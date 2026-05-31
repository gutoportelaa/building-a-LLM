import os
import time
import base64
import json
import urllib.request
import fitz
from pathlib import Path

def pymupdf_extraction(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def qwen3vl_extraction(pdf_path):
    # Get first page as image
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=150)
    img_data = pix.tobytes("png")
    b64_img = base64.b64encode(img_data).decode("utf-8")

    
    # Send to Ollama
    payload = {
        "model": "llama3.2-vision",
        "prompt": "Transcreva exatamente o texto desta imagem para o formato Markdown. Se houver tabelas, utilize a estrutura de tabelas do Markdown (| Coluna 1 | Coluna 2 |). Ignore repetições de cabeçalhos e rodapés de paginação.",
        "images": [b64_img],
        "stream": False
    }
    
    try:
        req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result.get("response", "")
    except Exception as e:
        return f"Erro Ollama (verifique se qwen2.5-vl está baixado e rodando): {e}"

def marker_extraction(pdf_path):
    import subprocess
    output_dir = "dados_benchmark/marker_out"
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["marker_single", pdf_path, "--output_dir", output_dir]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Find the generated md file
        base_name = Path(pdf_path).stem
        md_file = Path(output_dir) / base_name / f"{base_name}.md"
        if md_file.exists():
            return md_file.read_text(encoding='utf-8')
        return "Erro: Arquivo MD não encontrado após rodar marker."
    except Exception as e:
        return f"Erro Marker CLI: {e}"

def main():
    bench_dir = Path("dados_benchmark")
    if not bench_dir.exists():
        print(f"Diretório {bench_dir} não encontrado.")
        return
    
    pdfs = list(bench_dir.glob("*.pdf"))
    print(f"Iniciando benchmark em {len(pdfs)} arquivos...")
    
    results = []
    
    for pdf in pdfs:
        print(f"\n--- Analisando {pdf.name} ---")
        pdf_path = str(pdf)
        
        # PyMuPDF
        t0 = time.time()
        md_pymupdf = pymupdf_extraction(pdf_path)
        t_pymupdf = time.time() - t0
        print(f"PyMuPDF: {t_pymupdf:.2f}s")
        Path(bench_dir / f"{pdf.stem}_pymupdf.md").write_text(md_pymupdf, encoding="utf-8")
        
        # Marker
        t0 = time.time()
        md_marker = marker_extraction(pdf_path)
        t_marker = time.time() - t0
        print(f"Marker: {t_marker:.2f}s")
        Path(bench_dir / f"{pdf.stem}_marker.md").write_text(md_marker, encoding="utf-8")
        
        # Qwen-VL
        t0 = time.time()
        md_qwen = qwen3vl_extraction(pdf_path)
        t_qwen = time.time() - t0
        print(f"Qwen-VL: {t_qwen:.2f}s")
        Path(bench_dir / f"{pdf.stem}_qwenvl.md").write_text(md_qwen, encoding="utf-8")
        
        results.append({
            "arquivo": pdf.name,
            "tempo_pymupdf": t_pymupdf,
            "tempo_marker": t_marker,
            "tempo_qwenvl": t_qwen
        })
        
    print("\n" + "="*50)
    print("RESUMO DO BENCHMARK")
    print("="*50)
    print(f"{'Arquivo':<15} | {'PyMuPDF':<10} | {'Marker':<10} | {'Qwen-VL':<10}")
    print("-"*50)
    for r in results:
        print(f"{r['arquivo'][:10]}... | {r['tempo_pymupdf']:<8.2f}s | {r['tempo_marker']:<8.2f}s | {r['tempo_qwenvl']:<8.2f}s")

if __name__ == "__main__":
    main()
