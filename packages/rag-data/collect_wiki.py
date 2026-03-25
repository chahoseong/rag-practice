import os
import wikipediaapi
import json
import argparse
import time
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError

wiki = wikipediaapi.Wikipedia(
    language='ko',
    user_agent='rag-data-bot/0.1 (contact: your-email@example.com; research purposes)'
)

def safe_get_text(page, max_retries=3, delay=1):
    """API 호출 실패 시 재시도를 포함하여 안전하게 텍스트를 가져옵니다."""
    for i in range(max_retries):
        try:
            # text 접근 시 실제 네트워크 요청이 발생할 수 있음
            return page.text
        except (RequestsJSONDecodeError, Exception) as e:
            if i < max_retries - 1:
                print(f"⚠️ '{page.title}' 가져오기 실패 (시도 {i+1}/{max_retries}). {delay}초 후 재시도... Error: {e}")
                time.sleep(delay * (i + 1))
            else:
                print(f"❌ '{page.title}' 최종 가져오기 실패: {e}")
    return ""

def get_pages_in_category(cat_title: str, max_depth=2):
    category = wiki.page(cat_title)
    visited_cats = set()
    visited_titles = set()
    results = []

    def recurse(cat, depth):
        if depth > max_depth or cat.title in visited_cats:
            return
        visited_cats.add(cat.title)

        for member in cat.categorymembers.values():
            # API 부하 분산을 위한 아주 짧은 지연
            time.sleep(0.05)
            
            if member.ns == wikipediaapi.Namespace.CATEGORY:
                recurse(member, depth + 1)
            elif member.ns == wikipediaapi.Namespace.MAIN:
                if member.title not in visited_titles:
                    text = safe_get_text(member)
                    if len(text) > 500:
                        visited_titles.add(member.title)
                        results.append(member)
                        if len(results) % 10 == 0:
                            print(f"len(results): {len(results)}")

    recurse(category, 0)
    print(f"final len(results): {len(results)}")
    return results

def save_pages_as_jsonl(pages, output_path="data/wiki.jsonl"):
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "wiki.jsonl")
    elif not output_path.endswith(".jsonl"):
        output_path = f"{output_path}.jsonl"
    
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            # 이미 safe_get_text로 가져왔으므로 캐시된 데이터를 사용하게 됨
            text = page.text.strip()
            if not text: # 만약 비어있다면 다시 시도
                text = safe_get_text(page).strip()
                
            record = {
                "title": page.title,
                "url": page.fullurl,
                "text": text,
                "source_type": "Wikipedia"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ JSONL 저장 완료: {len(pages)}개 문서 → {output_path}")

def save_single_page_as_jsonl(page, output_path="data/wiki_single.jsonl"):
    if not page.exists():
        print(f"❌ 문서 '{page.title}' 가 존재하지 않습니다.")
        return
    
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "wiki_single.jsonl")
    elif not output_path.endswith(".jsonl"):
        output_path = f"{output_path}.jsonl"
        
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        record = {
            "title": page.title,
            "url": page.fullurl,
            "text": page.text.strip(),
            "source_type": "Wikipedia"
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅ 단일 문서 저장 완료: {page.title} → {output_path}")

def main():
    p = argparse.ArgumentParser(description="Collect Wikipedia pages by category or single page")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--category", type=str,
                       help="Category title to collect pages from. ex) '분류:조선_세종'")
    group.add_argument("--page", type=str,
                       help="Single page title to collect. ex) '세종대왕'")
    p.add_argument("--max_depth", type=int, default=3,
                   help="Maximum depth to traverse subcategories (category mode only)")
    p.add_argument("--output", type=str, default="data/wiki.jsonl",
                   help="Output JSONL file path")
    args = p.parse_args()

    if args.page:
        page = wiki.page(args.page)
        save_single_page_as_jsonl(page, args.output)
    else:
        pages = get_pages_in_category(args.category, args.max_depth)
        save_pages_as_jsonl(pages, args.output)

if __name__ == "__main__":
    main()
