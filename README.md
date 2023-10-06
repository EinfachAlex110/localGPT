# LocalGPT: Sichere, lokale Gespräche mit Ihren Dokumenten 🌐

**LocalGPT** ist eine Open-Source-Initiative, die es ermöglicht, mit Ihren Dokumenten zu kommunizieren, ohne Ihre Privatsphäre zu gefährden. Da alles lokal ausgeführt wird, können Sie sicher sein, dass keine Daten Ihren Computer verlassen. Tauchen Sie ein in die Welt sicherer, lokaler Dokumenteninteraktionen mit LocalGPT.

## Eigenschaften 🌟
- **Höchste Privatsphäre**: Ihre Daten bleiben auf Ihrem Computer und gewährleisten 100% Sicherheit.
- **Vielseitige Modellunterstützung**: Integrieren Sie nahtlos eine Vielzahl von Open-Source-Modellen, einschließlich HF, GPTQ, GGML und GGUF.
- **Vielfältige Einbettungen**: Wählen Sie aus einer Reihe von Open-Source-Einbettungen.
- **Wiederverwendung Ihres LLM**: Nach dem Herunterladen können Sie Ihr LLM wieder verwenden, ohne es erneut herunterladen zu müssen.
- **Chat-Verlauf**: Erinnert sich an Ihre vorherigen Gespräche (in einer Sitzung).
- **API**: LocalGPT verfügt über eine API, die Sie zum Erstellen von RAG-Anwendungen verwenden können.
- **Grafische Benutzeroberfläche**: LocalGPT wird mit zwei GUIs geliefert, eine verwendet die API und die andere ist eigenständig (basierend auf Streamlit).
- **GPU, CPU & MPS-Unterstützung**: Unterstützt mehrere Plattformen Out of the Box. Unterhalten Sie sich mit Ihren Daten unter Verwendung von `CUDA`, `CPU` oder `MPS` und mehr!

## Tauchen Sie tiefer ein mit unseren Videos 🎥
- [Detaillierte Code-Durchsicht](https://youtu.be/MlyoObdIHyo)
- [Llama-2 mit LocalGPT](https://youtu.be/lbFmceo4D5E)
- [Chat-Verlauf hinzufügen](https://youtu.be/d7otIM_MCZs)
- [LocalGPT - Aktualisiert (17.09.2023)](https://youtu.be/G_prHSKX9d4)

## Technische Details 🛠️
Durch die Auswahl der richtigen lokalen Modelle und die Kraft von `LangChain` können Sie die gesamte RAG-Pipeline lokal ausführen, ohne dass Daten Ihre Umgebung verlassen, und das mit akzeptabler Leistung.

- `ingest.py` verwendet `LangChain`-Werkzeuge, um das Dokument zu analysieren und Einbettungen lokal mit `InstructorEmbeddings` zu erstellen. Anschließend wird das Ergebnis in einer lokalen Vektordatenbank mit `Chroma` Vektor-Speicher gespeichert.
- `run_localGPT.py` verwendet ein lokales LLM, um Fragen zu verstehen und Antworten zu erstellen. Der Kontext für die Antworten wird aus dem lokalen Vektor-Speicher extrahiert, indem eine Ähnlichkeitssuche durchgeführt wird, um den richtigen Kontextteil aus den Dokumenten zu finden.
- Sie können dieses lokale LLM durch jedes andere LLM von HuggingFace ersetzen. Stellen Sie sicher, dass das LLM, das Sie auswählen, im HF-Format ist.

Dieses Projekt wurde inspiriert vom Original [privateGPT](https://github.com/imartinez/privateGPT).

## Gebaut mit 🧩
- [LangChain](https://github.com/hwchase17/langchain)
- [HuggingFace LLMs](https://huggingface.co/models)
- [InstructorEmbeddings](https://instructor-embedding.github.io/)
- [LLAMACPP](https://github.com/abetlen/llama-cpp-python)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

# Umgebungseinrichtung 🌍

1. 📥 Klone das Repo mit git:

```shell
git clone https://github.com/PromtEngineer/localGPT.git
```

2. 🐍 Installiere [conda](https://www.anaconda.com/download) für die Verwaltung virtueller Umgebungen. Erstelle und aktiviere eine neue virtuelle Umgebung.

```shell
conda create -n localGPT python=3.10.0
conda activate localGPT
```

3. 🛠️ Installiere die Abhängigkeiten mit pip

Richten Sie Ihre Umgebung ein, um den Code auszuführen, und installieren Sie zunächst alle Anforderungen:

```shell
pip install -r requirements.txt
```

***LLAMA-CPP installieren:***

LocalGPT verwendet [LlamaCpp-Python](https://github.com/abetlen/llama-cpp-python) für GGML (Sie benötigen llama-cpp-python <=0.1.76) und GGUF (llama-cpp-python >=0.1.83) Modelle.

Falls Sie BLAS oder Metal mit [llama-cpp](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal) verwenden möchten, können Sie entsprechende Flags setzen:

Für `NVIDIA` GPUs-Unterstützung verwenden Sie `cuBLAS`

```shell
# Beispiel: cuBLAS
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

Für Apple Metal (`M1/M2`) Unterstützung verwenden Sie

```shell
# Beispiel: METAL
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```
Weitere Details finden Sie unter [llama-cpp](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal).

## Docker 🐳

Das Installieren der benötigten Pakete für GPU-Inferenzen auf NVIDIA-GPUs, wie gcc 11 und CUDA 11, kann zu Konflikten mit anderen Paketen in Ihrem System führen.
Als Alternative zu Conda können Sie Docker mit der bereitgestellten Dockerfile verwenden.
Es beinhaltet CUDA, Ihr System benötigt nur Docker, BuildKit, Ihren NVIDIA GPU-Treiber und das NVIDIA Container Toolkit.
Bauen als `docker build . -t localgpt`, erfordert BuildKit.
Docker BuildKit unterstützt GPU während der *docker build*-Zeit derzeit nicht, nur während der *docker run*.
Ausführen als `docker run -it --mount src="$HOME/.cache",target=/root/.cache,type=bind --gpus=all localgpt`.

## Testdatensatz



Zum Testen wird dieses Repository mit der [Verfassung der USA](https://constitutioncenter.org/media/files/constitution.pdf) als Beispiel-Datei geliefert.

## Ihre EIGENEN Daten einpflegen.
Legen Sie Ihre Dateien in den Ordner `SOURCE_DOCUMENTS`. Sie können mehrere Ordner innerhalb des Ordners `SOURCE_DOCUMENTS` anlegen und der Code wird Ihre Dateien rekursiv lesen.

### Unterstützte Dateiformate:
LocalGPT unterstützt derzeit die folgenden Dateiformate. LocalGPT verwendet `LangChain` zum Laden dieser Dateiformate. Der Code in `constants.py` verwendet ein `DOCUMENT_MAP`-Wörterbuch, um ein Dateiformat dem entsprechenden Loader zuzuordnen. Um die Unterstützung für ein weiteres Dateiformat hinzuzufügen, fügen Sie einfach dieses Wörterbuch mit dem Dateiformat und dem entsprechenden Loader aus [LangChain](https://python.langchain.com/docs/modules/data_connection/document_loaders/) hinzu.

```shell
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
```

### Ingest

Führen Sie den folgenden Befehl aus, um alle Daten einzupflegen.

Wenn Sie `cuda` auf Ihrem System eingerichtet haben.

```shell
python ingest.py
```
Sie werden eine Ausgabe wie diese sehen:
<img width="1110" alt="Screenshot 2023-09-14 at 3 36 27 PM" src="https://github.com/PromtEngineer/localGPT/assets/134474669/c9274e9a-842c-49b9-8d95-606c3d80011f">


Verwenden Sie das Argument `device_type`, um einen bestimmten Gerätetyp anzugeben.
Zum Ausführen auf `cpu`

```sh
python ingest.py --device_type cpu
```

Zum Ausführen auf `M1/M2`

```sh
python ingest.py --device_type mps
```

Verwenden Sie `help` für eine vollständige Liste der unterstützten Geräte.

```sh
python ingest.py --help
```

Dies wird einen neuen Ordner namens `DB` erstellen und ihn für den neu erstellten Vektor-Speicher verwenden. Sie können so viele Dokumente einpflegen, wie Sie möchten, und alle werden in der lokalen Einbettungsdatenbank angesammelt.
Wenn Sie von einer leeren Datenbank aus starten möchten, löschen Sie die `DB` und nehmen Sie die Einpflege Ihrer Dokumente erneut vor.

Hinweis: Wenn Sie dies zum ersten Mal ausführen, benötigen Sie Internetzugang, um das Einbettungsmodell herunterzuladen (Standard: `Instructor Embedding`). In den folgenden Läufen werden keine Daten Ihre lokale Umgebung verlassen und Sie können Daten ohne Internetverbindung einpflegen.

## Stellen Sie Ihren Dokumenten Fragen, lokal!

Um mit Ihren Dokumenten zu chatten, führen Sie den folgenden Befehl aus (standardmäßig wird es auf `cuda` ausgeführt).

```shell
python run_localGPT.py
```
Sie können auch den Gerätetyp angeben, genau wie `ingest.py`

```shell
python run_localGPT.py --device_type mps # zum Ausführen auf Apple Silicon
```

Dies wird den eingepflegten Vektor-Speicher und das Einbettungsmodell laden. Ihnen wird eine Aufforderung präsentiert:

```shell
> Geben Sie eine Abfrage ein:
```

Nachdem Sie Ihre Frage eingegeben haben, drücken Sie die Eingabetaste. LocalGPT wird je nach Ihrer Hardware einige Zeit in Anspruch nehmen. Sie erhalten eine Antwort wie die untenstehende.
<img width="1312" alt="Screenshot 2023-09-14 at 3 33 19 PM" src="https://github.com/PromtEngineer/localGPT/assets/134474669/a7268de9-ade0-420b-a00b-ed12207dbe41">

Sobald die Antwort generiert wurde, können Sie eine weitere Frage stellen, ohne das Skript erneut auszuführen. Warten Sie einfach auf die Aufforderung.

***Hinweis:*** Wenn Sie dies zum ersten Mal ausführen, benötigen Sie eine Internetverbindung, um das LLM herunterzuladen (Standard: `TheBloke/Llama-2-7b-Chat-GGUF`). Danach können Sie Ihre Internetverbindung ausschalten und die Skriptinferenz würde immer noch funktionieren. Es gelangen keine Daten aus Ihrer lokalen Umgebung heraus.

Geben Sie `exit` ein, um das Skript zu beenden.

### Zusätzliche Optionen mit run_localGPT.py

Sie können die Flag `--show_sources` mit `run_localGPT.py` verwenden, um anzuzeigen, welche Blöcke vom Einbettungsmodell abgerufen wurden. Standardmäßig werden 4 verschiedene Quellen/Blöcke angezeigt. Sie können die Anzahl der Quellen/Blöcke ändern

```shell
python run_localGPT.py --show_sources
```

Eine andere Option ist die Aktivierung des Chat-Verlaufs. ***Hinweis***: Dies ist standardmäßig deaktiviert und kann durch Verwendung der Flag `--use_history` aktiviert werden. Das Kontextfenster ist begrenzt, so dass das Aktivieren des Verlaufs es verwenden und möglicherweise überlaufen kann.

```shell
python run_localGPT.py --use_history
```

# Starten der grafischen Benutzeroberfläche

1. Öffnen

 Sie `constants.py` in einem Editor Ihrer Wahl und fügen Sie je nach Wahl das LLM hinzu, das Sie verwenden möchten. Standardmäßig wird das folgende Modell verwendet:

   ```shell
   MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
   MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
   ```

3. Öffnen Sie ein Terminal und aktivieren Sie Ihre Python-Umgebung, die die in requirements.txt installierten Abhängigkeiten enthält.

4. Navigieren Sie zum Verzeichnis `/LOCALGPT`.

5. Führen Sie den folgenden Befehl aus: `python run_localGPT_API.py`. Die API sollte zu laufen beginnen.

6. Warten Sie, bis alles geladen ist. Sie sollten so etwas sehen wie `INFO:werkzeug:Press CTRL+C to quit`.

7. Öffnen Sie ein zweites Terminal und aktivieren Sie dieselbe Python-Umgebung.

8. Navigieren Sie zum Verzeichnis `/LOCALGPT/localGPTUI`.

9. Führen Sie den Befehl `python localGPTUI.py` aus.

10. Öffnen Sie einen Webbrowser und gehen Sie zur Adresse `http://localhost:5111/`.


# Wie wählt man unterschiedliche LLM-Modelle aus?

Um die Modelle zu ändern, müssen Sie sowohl `MODEL_ID` als auch `MODEL_BASENAME` festlegen.

1. Öffnen Sie `constants.py` in dem Editor Ihrer Wahl.
2. Ändern Sie `MODEL_ID` und `MODEL_BASENAME`. Wenn Sie ein quantisiertes Modell verwenden (`GGML`, `GPTQ`, `GGUF`), müssen Sie `MODEL_BASENAME` angeben. Bei nicht quantisierten Modellen setzen Sie `MODEL_BASENAME` auf `NONE`
5. Es gibt eine Reihe von Beispielmodellen von HuggingFace, die bereits getestet wurden, um mit dem ursprünglich trainierten Modell (Endung mit HF oder haben eine .bin in seinen "Dateien und Versionen") und quantisierten Modellen (Endung mit GPTQ oder haben eine .no-act-order oder .safetensors in seinen "Dateien und Versionen") ausgeführt zu werden.
6. Für Modelle, die mit HF enden oder eine .bin in ihren "Dateien und Versionen" auf ihrer HuggingFace-Seite haben.

   - Stellen Sie sicher, dass Sie ein `MODEL_ID` ausgewählt haben. Zum Beispiel -> `MODEL_ID = "TheBloke/guanaco-7B-HF"`
   - Gehen Sie zur [HuggingFace Repo](https://huggingface.co/TheBloke/guanaco-7B-HF)

7. Für Modelle, die GPTQ in ihrem Namen enthalten und/oder eine .no-act-order oder .safetensors-Erweiterung in ihren "Dateien und Versionen" auf ihrer HuggingFace-Seite haben.

   - Stellen Sie sicher, dass Sie ein `MODEL_ID` ausgewählt haben. Zum Beispiel -> model_id = `"TheBloke/wizardLM-7B-GPTQ"`
   - Gehen Sie zur entsprechenden [HuggingFace Repo](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) und wählen Sie "Dateien und Versionen".
   - Wählen Sie einen der Modellnamen aus und setzen Sie ihn als `MODEL_BASENAME`. Zum Beispiel -> `MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"`

8. Folgen Sie den gleichen Schritten für `GGUF` und `GGML`-Modelle.

# GPU- und VRAM-Anforderungen

Unten finden Sie die VRAM-Anforderungen für verschiedene Modelle, abhängig von ihrer Größe (Milliarden von Parametern). Die Schätzungen in der Tabelle beinhalten nicht den VRAM, der von den Einbettungsmodellen verwendet wird - diese verwenden zusätzlich 2 GB - 7 GB VRAM, abhängig vom Modell.

| Modellgröße (B) | float32   | float16   | GPTQ 8bit      | GPTQ 4bit          |
| ------- | --------- | --------- | -------------- | ------------------ |
| 7B      | 28 GB     | 14 GB     | 7 GB - 9 GB    | 3,5 GB - 5 GB      |
| 13B     | 52 GB     | 26 GB     | 13 GB - 15 GB  | 6,5 GB - 8 GB      |
| 32B     | 130 GB    | 65 GB     | 32,5 GB - 35 GB| 16,25 GB - 19 GB   |
| 65B     | 260,8 GB  | 130,4 GB  | 65,2 GB - 67 GB| 32,6 GB - 35 GB    |


# Systemanforderungen

## Python-Version

Um diese Software zu verwenden, müssen Sie Python 3.10 oder eine neuere Version installiert haben. Frühere Versionen von Python werden nicht kompiliert.

## C++ Compiler

Wenn Sie während des `pip install`-Prozesses einen Fehler beim Erstellen eines Rades erhalten, müssen Sie möglicherweise einen C++-Compiler auf Ihrem Computer installieren.

### Für Windows 10/11

Um einen C++-Compiler unter Windows 10/11 zu installieren, gehen Sie wie folgt vor:

1. Installieren Sie Visual Studio 2022.
2. Stellen Sie sicher, dass die folgenden Komponenten ausgewählt sind:
   - Universal Windows Platform development
   - C++ CMake tools für Windows
3. Laden Sie den MinGW-Installer von der [MinGW-Website](https://sourceforge.net/projects/mingw/) herunter.
4. Führen Sie den Installer aus und wählen Sie die Komponente "gcc" aus.

### NVIDIA-Treiberprobleme:

Folgen Sie dieser [Seite](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04), um NVIDIA-Treiber zu installieren.

## Sternenhistorie

[![Sternenhistorie-Diagramm](https://api.star-history.com/svg?repos=PromtEngineer/localGPT&type=Date)](https://star-history.com/#PromtEngineer/localGPT&Date)

# Haftungsausschluss

Dies ist ein Testprojekt zur Überprüfung der Machbarkeit einer vollständig lokalen Lösung für die Beantwortung von Fragen mit LLMs und Vektoreinbettungen. Es ist nicht produktionsbereit und nicht für den Einsatz in der Produktion vorgesehen. Vicuna-7B basiert auf dem Llama-Modell, sodass es die ursprüngliche Llama-Lizenz hat.

# Häufig
