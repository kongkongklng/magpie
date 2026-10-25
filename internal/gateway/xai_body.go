package gateway

import (
	"bytes"
	"encoding/json"

	"github.com/yetone/magpie/internal/provider"
)

// xAI's own API (api.x.ai), wherever a Grok model is served, takes two
// things a client's wire puts in wrong, as the Grok subscription's plugin
// shapes its bodies too (provider's grokShaped):
//
//   - Codex hands a sealed reasoning item back with "content": null
//     beside its encrypted_content, which the backend answers "Could not
//     decode the compaction blob. Ensure it is unmodified from the compact
//     response." (#3392, kongkongklng: the first turn went through, every
//     one after it 400ed) — the null goes.
//   - Claude Code sends a system turn mid-conversation (Anthropic made
//     them GA), which xAI's Messages answers "Invalid request content:
//     Invalid message role." (#3392: every claude call 400ed) — each folds
//     into the top-level system, the only place it takes the role.
//
// The relay and the translation shape alike: a refusal of custom tools
// puts a Codex request on the translated path from the next turn on
// (custom_tool.go), and there is nothing to learn from either answer —
// xAI takes no null content and no system turn at all.

// xaiBody shapes body as xAI's own API takes it on proto.
func xaiBody(p provider.Provider, proto provider.Protocol, body []byte) []byte {
	if p.Host() != "api.x.ai" {
		return body
	}
	switch proto {
	case provider.Responses:
		return withoutNullReasoningContent(body)
	case provider.Anthropic:
		return systemTurnsFolded(body)
	}
	return body
}

// withoutNullReasoningContent is a Responses body with the null content of
// a reasoning item gone (#3392).
func withoutNullReasoningContent(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"reasoning"`)) || !bytes.Contains(body, []byte(`"content"`)) {
		return body
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var m map[string]any
	if dec.Decode(&m) != nil || m == nil {
		return body
	}
	dirty := false
	items, _ := m["input"].([]any)
	for _, it := range items {
		im, _ := it.(map[string]any)
		if im == nil || im["type"] != "reasoning" {
			continue
		}
		if c, there := im["content"]; there && c == nil {
			delete(im, "content")
			dirty = true
		}
	}
	if !dirty {
		return body
	}
	out, err := marshalPlain(m)
	if err != nil {
		return body
	}
	return out
}

// systemTurnsFolded is an Anthropic body with each system turn in messages
// folded into the top-level system, its text and blocks as they came
// (#3392).
func systemTurnsFolded(body []byte) []byte {
	if !bytes.Contains(body, []byte(`"messages"`)) || !bytes.Contains(body, []byte(`"system"`)) {
		return body
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var m map[string]any
	if dec.Decode(&m) != nil || m == nil {
		return body
	}
	msgs, _ := m["messages"].([]any)
	kept := msgs[:0:0]
	var folded []any // each system turn's content, a string or blocks
	for _, mm := range msgs {
		im, _ := mm.(map[string]any)
		if im == nil || im["role"] != "system" {
			kept = append(kept, mm)
			continue
		}
		switch c := im["content"].(type) {
		case string:
			folded = append(folded, c)
		case []any:
			folded = append(folded, c...)
		}
	}
	if len(kept) == len(msgs) {
		return body
	}
	m["messages"] = kept
	// the system prompt: what was there, then each turn's text and blocks
	var blocks []any
	switch s := m["system"].(type) {
	case string:
		if s != "" {
			blocks = append(blocks, map[string]any{"type": "text", "text": s})
		}
	case []any:
		blocks = append(blocks, s...)
	}
	for _, f := range folded {
		if s, ok := f.(string); ok {
			if s != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": s})
			}
			continue
		}
		blocks = append(blocks, f)
	}
	if len(blocks) > 0 {
		m["system"] = blocks
	} else {
		delete(m, "system")
	}
	out, err := marshalPlain(m)
	if err != nil {
		return body
	}
	return out
}
