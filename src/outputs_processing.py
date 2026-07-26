import os
import json
import zhconv

def main():
    # Define file paths
    eval_results_path = os.path.join("eval-results", "sgf_rag_multi-agentic-llm.json")
    buju_docs_path = os.path.join("source-doc", "buju", "圍棋佈局解說.json")
    output_path = os.path.join("eval-results", "output_processed.json")

    print(f"Reading generated results from: {eval_results_path}")
    if not os.path.exists(eval_results_path):
        print(f"Error: {eval_results_path} does not exist.")
        return

    print(f"Reading original commentaries from: {buju_docs_path}")
    if not os.path.exists(buju_docs_path):
        print(f"Error: {buju_docs_path} does not exist.")
        return

    # Load data
    with open(eval_results_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    with open(buju_docs_path, 'r', encoding='utf-8') as f:
        buju_data = json.load(f)

    # Create mapping of name -> commentary (translated to Traditional Chinese)
    commentary_map = {}
    for item in buju_data:
        name = item.get("name")
        commentary = item.get("commentary", "")
        if name and commentary:
            # Convert commentary to Traditional Chinese (TW variant)
            commentary_tw = zhconv.convert(commentary, 'zh-tw')
            commentary_map[name] = commentary_tw

    # Process and merge entries
    processed_results = []
    for entry in eval_data:
        q_id = entry.get("question_id")
        joseki_zh = entry.get("joseki_sequence_human_zh", "")
        final_output = entry.get("final_localized_output", "")
        
        # Match with commentary
        original_commentary = commentary_map.get(q_id, "")
        
        # Build dictionary with specific field order: name, joseki, final, commentary
        processed_entry = {
            "name": q_id,
            "joseki_sequence_human_zh": joseki_zh,
            "final_localized_output": final_output,
            "commentary": original_commentary
        }
        processed_results.append(processed_entry)

    # Save to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(processed_results, outfile, ensure_ascii=False, indent=2)

    print(f"Successfully integrated results and saved to: {output_path}")
    print(f"Processed {len(processed_results)} entries.")

if __name__ == "__main__":
    main()
