import os
import sys

# Permite importar o pacote de desenvolvimento local
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bindings/python/python")))

import graphite_db as graphite

db_path = os.path.join(os.path.dirname(__file__), "codigo_penal.graph")
md_path = os.path.join(os.path.dirname(__file__), "codigo_penal.md")

# 1. Ingestao automatica apenas se o banco ainda nao existir no disco
if not os.path.exists(db_path):
    print("Banco nao encontrado. Ingerindo codigo_penal.md pela primeira vez...")
    db = graphite.open(db_path, dim=384, max_tokens=1500)
    total = db.ingest(md_path)
    db.close()
    print(f"Ingestao concluida: {total} secoes gravadas no disco!\n")

# 2. Abre o banco persistido do disco instantaneamente
db = graphite.open(db_path, dim=384, max_tokens=1500)

# 3. Consulta e impressao do resultado
pergunta = "O que diz o artigo 121 do codigo penal sobre homicidio?"
print(f"Pergunta: \"{pergunta}\"\n")
print("Resultado:")
print("-" * 70)

resultado = db.query(pergunta, top_k=3, max_tokens=1500)
print(resultado.markdown)

print("-" * 70)
print(f"Tokens: {resultado.token_count}")
