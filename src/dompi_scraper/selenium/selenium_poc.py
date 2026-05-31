import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

# ==============================================================================
# CONFIGURAÇÕES DA POC
# ==============================================================================
BASE_URL = "https://www.diarioficialdosmunicipios.org/consulta/ConPublicacaoGeral/ConPublicacaoGeral.php"
MUNICIPIO_ALVO = "Campo Maior"
ENTIDADE_ALVO = "Camara"

def main():
    print("Iniciando Prova de Conceito com Selenium...")
    
    # 1. Início da contagem de tempo para o relatório
    start_time = time.time()
    
    # Inicializa o navegador (Chrome)
    # Nota: Em versões recentes do Selenium (>= 4.6), não é necessário baixar o chromedriver manualmente.
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # Descomente para rodar sem abrir a janela visual
    driver = webdriver.Chrome(options=options)
    
    try:
        # 2. Acessa a página principal
        print(f"Acessando: {BASE_URL}")
        driver.get(BASE_URL)
        
        # Aguarda os elementos do formulário carregarem
        wait = WebDriverWait(driver, 15)
        
        # 3. Preenche o formulário (Interação Dinâmica)
        print(f"Selecionando Município: {MUNICIPIO_ALVO}")
        select_mun = Select(wait.until(EC.presence_of_element_located((By.NAME, "nomemunicipio"))))
        select_mun.select_by_visible_text(MUNICIPIO_ALVO)
        
        print(f"Selecionando Entidade: {ENTIDADE_ALVO}")
        select_ent = Select(driver.find_element(By.NAME, "nomeentidade"))
        select_ent.select_by_visible_text(ENTIDADE_ALVO)
        
        # 4. Aciona a pesquisa
        # O portal DOM-PI usa links/botões embutidos para submeter o form Scriptcase
        print("Clicando em Pesquisar...")
        btn_pesquisa = driver.find_element(By.XPATH, "//a[contains(text(), 'Pesquisa Avançada') or contains(@onclick, 'pesq')]")
        btn_pesquisa.click()
        
        # 5. Lida com o Iframe (Comportamento descoberto na engenharia reversa do Requests)
        # O Scriptcase frequentemente joga os resultados para dentro de um iframe
        print("Aguardando carregamento dos resultados (iframe)...")
        time.sleep(3) # Pausa estática para garantir o processamento do servidor legad0
        
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        if iframes:
            driver.switch_to.frame(iframes[0])
            
        # 6. Extração dos links dos PDFs
        print("Buscando links de documentos na tabela de resultados...")
        # Localiza as tags <a> que contêm a chamada JavaScript para o PDF
        pdf_elements = wait.until(EC.presence_of_all_elements_located((
            By.XPATH, "//a[contains(@href, 'javascript:nm_gp_submit5') or contains(@href, '.pdf')]"
        )))
        
        print(f"\n--- ENCONTRADOS {len(pdf_elements)} DOCUMENTOS NA PÁGINA 1 ---")
        for idx, elem in enumerate(pdf_elements[:5]): # Mostrando apenas os 5 primeiros para a PoC
            href = elem.get_attribute("href")
            print(f"Doc {idx+1}: {href}")
            
    except Exception as e:
        print(f"Ocorreu um erro durante a execução: {e}")
        
    finally:
        # 7. Finalização e coleta de métricas
        driver.quit()
        end_time = time.time()
        tempo_total = end_time - start_time
        print("\n" + "="*50)
        print("MÉTRICAS PARA O RELATÓRIO:")
        print(f"Tempo total de execução (1 página): {tempo_total:.2f} segundos")
        print("="*50)

if __name__ == "__main__":
    main()