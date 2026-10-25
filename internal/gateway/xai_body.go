package gateway

import (
	"bytes"
	"encoding/json"

	"github.com/yetone/magpie/internal/provider"
)

// xAI's own API (api.x.ai), wherever a Grok model is served, takes three
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
//   - Claude Code offers tools whose root schema has no required field
//     (CronList and EnterPlanMode among them), which xAI's validator reads
//     as null and turns away with "/required: null is not of type array" —
//     an empty array goes in, which requires nothing, as the missing field
//     meant.
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
		body = withoutNullReasoningContent(body)
	case provider.Anthropic:
		body = systemTurnsFolded(body)
	}
	return requiredArrays(body, proto)
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

// requiredArrays is body with an empty required array on each tool schema
// whose root has none (#3392). xAI's validator reads the missing field as
// null and refuses the request.
func requiredArrays(body []byte, proto provider.Protocol) []byte {
	if !bytes.Contains(body, []byte(`"tools"`)) {
		return body
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var m map[string]any
	if dec.Decode(&m) != nil || m == nil {
		return body
	}
	tools, _ := m["tools"].([]any)
	if !fillRequiredArrays(proto, tools) {
		return body
	}
	out, err := marshalPlain(m)
	if err != nil {
		return body
	}
	return out
}

// fillRequiredArrays fills the root required array of each tool schema that
// has none, and reports whether it changed any.
func fillRequiredArrays(proto provider.Protocol, tools []any) bool {
	changed := false
	for _, t := range tools {
		tm, _ := t.(map[string]any)
		if tm == nil {
			continue
		}
		if proto == provider.Responses && tm["type"] == "namespace" {
			nested, _ := tm["tools"].([]any)
			if fillRequiredArrays(proto, nested) {
				changed = true
			}
			continue
		}
		var schema map[string]any
		switch proto {
		case provider.Anthropic:
			schema, _ = tm["input_schema"].(map[string]any)
		case provider.Responses:
			schema, _ = tm["parameters"].(map[string]any)
		}
		if schema == nil {
			continue
		}
		if required, there := schema["required"]; !there || required == nil {
			schema["required"] = []any{}
			changed = true
		}
	}
	return changed
}
