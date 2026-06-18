#!/usr/bin/env python3
"""Valida os nomes de município do to-do contra o formulário ao vivo do DOM-PI.
1 requisição (load_form_context). Não baixa nada, não raspa publicações."""
import sys
sys.path.insert(0, "src")
from dompi_scraper.pipeline import build_session, load_form_context, _normalize_key

TERRITORIOS = {
    "cocais": ["Barras","Batalha","Brasileira","Campo Largo do Piauí","Domingos Mourão","Esperantina","Joaquim Pires","Joca Marques","Lagoa de São Francisco","Luzilândia","Madeiro","Matias Olímpio","Milton Brandão","Morro do Chapéu do Piauí","Nossa Senhora dos Remédios","Pedro II","Piracuruca","Piripiri","Porto","São João da Fronteira","São João do Arraial","São José do Divino"],
    "vale_do_rio_guaribas": ["Alagoinha do Piauí","Alegrete do Piauí","Aroeiras do Itaim","Bocaina","Campo Grande do Piauí","Dom Expedito Lopes","Francisco Santos","Fronteiras","Geminiano","Itainópolis","Monsenhor Hipólito","Paquetá","Picos","Pio IX","Santana do Piauí","Santo Antônio de Lisboa","São João da Canabrava","São José do Piauí","São Julião","São Luís do Piauí","Sussuapara","Vera Mendes","Vila Nova do Piauí"],
    "chapada_vale_do_rio_itaim": ["Acauã","Belém do Piauí","Betânia do Piauí","Caldeirão Grande do Piauí","Caridade do Piauí","Curral Novo do Piauí","Francisco Macedo","Jacobina do Piauí","Jaicós","Marcolândia","Massapê do Piauí","Padre Marcos","Patos do Piauí","Paulistana","Queimada Nova","Simões"],
    "vale_do_caninde": ["Bela Vista do Piauí","Cajazeiras do Piauí","Campinas do Piauí","Colônia do Piauí","Conceição do Canindé","Floresta do Piauí","Isaías Coelho","Oeiras","Santa Cruz do Piauí","Santa Rosa do Piauí","Santo Inácio do Piauí","São Francisco de Assis do Piauí","São Francisco do Piauí","São João da Varjota","Simplício Mendes","Tanque do Piauí","Wall Ferraz"],
    "serra_da_capivara": ["Anísio de Abreu","Bonfim do Piauí","Campo Alegre do Fidalgo","Capitão Gervásio Oliveira","Caracol","Coronel José Dias","Dirceu Arcoverde","Dom Inocêncio","Fartura do Piauí","Guaribas","João Costa","Jurema","Lagoa do Barro do Piauí","São Braz do Piauí","São João do Piauí","São Lourenço do Piauí","São Raimundo Nonato","Várzea Branca"],
    "mangabeiras": ["Alvorada do Gurguéia","Avelino Lopes","Barreiras do Piauí","Bom Jesus","Colônia do Gurguéia","Corrente","Cristalândia do Piauí","Cristino Castro","Currais","Curimatá","Eliseu Martins","Gilbués","Júlio Borges","Manoel Emídio","Monte Alegre do Piauí","Morro Cabeça no Tempo","Palmeira do Piauí","Parnaguá","Redenção do Gurguéia","Riacho Frio","Santa Filomena","Santa Luz","São Gonçalo do Gurguéia","Sebastião Barros"],
    "teresina": ["Teresina"],
    "parnaiba": ["Parnaíba"],
}

s = build_session()
ctx = load_form_context(s)
opts = ctx.get("municipio_options", {})
print(f"Formulário carregado: {len(opts)} chaves de município | erro={ctx.get('error') or 'nenhum'}")
if not opts:
    print("!! Sem opções de município — provável bloqueio de rede/site. Abortando.")
    sys.exit(1)


def resolver(nome):
    """Devolve (valor_site, como) ou (None, motivo). Cobre abreviação 'do piaui'->'do pi'
    e busca por radical (startswith do nome sem o sufixo 'do piaui'/'do pi')."""
    nk = _normalize_key(nome)
    if nk in opts:
        return opts[nk], "exato"
    nk2 = nk.replace("do piaui", "do pi")
    if nk2 in opts:
        return opts[nk2], "abrev-do-pi"
    stem = nk.replace(" do piaui", "").replace(" do pi", "")
    cands = sorted({v for k, v in opts.items() if k == stem or k.startswith(stem + " ")})
    if len(cands) == 1:
        return cands[0], "radical"
    if len(cands) > 1:
        return None, f"ambiguo:{cands}"
    return None, "nao-encontrado"

import json
resolvido_por_slug = {}
total_ok = total_miss = 0
for slug, muns in TERRITORIOS.items():
    valores, misses = [], []
    for m in muns:
        v, como = resolver(m)
        if v is not None:
            valores.append(v)
        else:
            misses.append((m, como))
    resolvido_por_slug[slug] = valores
    total_ok += len(valores); total_miss += len(misses)
    print(f"  {slug:<28} {len(valores)}/{len(muns)} resolvidos" + (f"  [{len(misses)} MISS]" if misses else "  [OK]"))
    for m, como in misses:
        # mostra candidatos do formulário que contêm o radical, p/ inspeção
        nk = _normalize_key(m)
        radicais = sorted({v for k, v in opts.items() if nk.split()[0] in k})[:6]
        print(f"        ✗ {m}  ({como})  candidatos~: {radicais}")

print(f"\nTOTAL: {total_ok} resolvidos, {total_miss} não encontrados")
print("\n=== LISTAS RESOLVIDAS (formato do site) ===")
print(json.dumps(resolvido_por_slug, ensure_ascii=False, indent=1))
