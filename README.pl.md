# foremankit

[English](README.md) · Polski

Przenośny zestaw do Claude Code, który oddaje tańszym lub lokalnym modelom pracę, której nie warto robić na limicie Claude Pro: boilerplate, testy, skrypty pandas/SQL, streszczenia dużych logów.

> foremankit to niezależny, nieoficjalny dodatek do rozszerzenia Claude Code (VS Code). Nie jest powiązany z firmą Anthropic, przez nią zatwierdzony ani sponsorowany.

**Analogia:** Claude to kierownik zmiany. Sam podejmuje decyzje i sprawdza wyniki, a robotę „na łopatę” zleca podwykonawcom. `foremankit` jest dyspozytorem: wie, który podwykonawca jest dziś wolny, ile kosztuje i komu wolno pokazać dane firmowe (tylko „swoim”, czyli Ollamie na własnym sprzęcie).

| Rola | Kto | Co robi |
|---|---|---|
| Nadzorca | główna sesja Claude Code (Sonnet/Opus) | planuje, weryfikuje, robi drobne poprawki |
| Doradca | subagent `opus-advisor` (Opus, tylko odczyt) | wołany po 2 nieudanych próbach albo przy decyzji architektonicznej |
| Przekaźnik | subagent `delegator` (Haiku) | woła MCP `delegate` i oddaje tylko ścieżkę do wyniku |
| Wykonawcy | DeepSeek, Gemini, Groq, OpenRouter, Mistral, (Cerebras), lokalny serwer LLM (llama.cpp, opcjonalnie Ollama) własny i na drugim laptopie | piszą kod i analizy |

Priorytety: 1) koszt i oszczędzanie limitu, 2) przenośność, 3) prywatność.

---

## Instalacja

Wymagania: Python 3.10+, Claude Code CLI (`claude`) w PATH. Działa na Windows, Linuksie i macOS.

```bash
cd foremankit
python install.py --dry-run     # pokazuje, co zostanie zrobione (bez zmian)
python install.py               # venv + paczka + agenci + reguły + hooki + rejestracja MCP
python install.py --check       # co z foremankita jest teraz zainstalowane (tylko odczyt)
```

`install.py` jest idempotentny, więc można go uruchamiać wielokrotnie. Robi kolejno:
1. tworzy `foremankit/.venv` i instaluje w nim paczkę (`pip install -e .[dev]`), bez żadnej instalacji globalnej;
2. przy pierwszym uruchomieniu zapisuje migawkę plików, które zmieni (`~/.config/foremankit/install-state/`), żeby dało się wrócić do stanu sprzed instalacji;
3. kopiuje `claude/agents/*.md` do `~/.claude/agents/`;
4. dokleja `claude/CLAUDE.global.md` do `~/.claude/CLAUDE.md` między znacznikami `<!-- BEGIN foremankit -->` i `<!-- END foremankit -->`;
5. rejestruje serwer MCP: `claude mcp add --scope user foremankit -- <venv>/Scripts/foremankit-mcp` (skrypt konsolowy, nie `python -m`, bo folder projektu o nazwie `foremankit/` przesłoniłby zainstalowaną paczkę);
6. dopisuje do `~/.claude/settings.json` cztery hooki (plik jest zapisywany bez BOM, bo parser Node'a odrzuca BOM; każdy hook kończy kodem 0 i nigdy nie blokuje pracy):
   - `PreToolUse` i `PostToolUse` na `mcp__foremankit__delegate` (`hooks/notify_delegate.py`): przed i po każdym wywołaniu pokazuje jednolinijkowe powiadomienie (`→ calling foremankit…`, `← response from…`), niezależnie od tego, czy model pamięta regułę z `CLAUDE.md`;
   - `UserPromptSubmit` (`hooks/session_length_reminder.py`; w sesjach na modelu lokalnym przy 85 % kontekstu nakazuje też zatrzymanie pracy i handoff, a sesji w chmurze nigdy nie zgłasza zapełnienia modelu lokalnego): gdy transkrypcja przekroczy 8 MB (`FOREMANKIT_SESSION_REMINDER_MB`), raz przypomina o nowej sesji (potem najwyżej co 90 min i po przyroście o 4 MB). Przy orkiestratorze lokalnym ostrzega też przy 60, 75 i 85 % okna kontekstu modelu;
   - `PostToolUse` na `mcp__foremankit__delegate` (`hooks/auto_checkpoint.py`): po zadaniu agenta zapisuje krótkie podsumowanie w `<projekt>/.agent-out/checkpoint-<czas>.md` i notuje je w `state.json`. Nic nie jest wysyłane na zewnątrz;
7. sprawdza, czy ustawiono `CLAUDE_CODE_SUBAGENT_MODEL` (nadpisuje pole `model` subagentów, więc **nie ustawiaj go**);
8. wypisuje, których kluczy brakuje.

Opcje instalacji:

| Opcja | Co robi |
|---|---|
| `--dry-run` | pokazuje każdą akcję, niczego nie zmienia (działa też z `--uninstall`) |
| `--skip-claude` | tylko venv i paczka; nie rusza niczego w `~/.claude` |
| `--local-orchestrator` | dodatkowo kieruje wtyczkę Claude Code przez bramkę do lokalnego Qwena (zmienia ustawienia VS Code; opis w sekcji „Lokalny orkiestrator”) |

Każdy modyfikowany plik w `~/.claude` jest najpierw kopiowany jako `*.bak-foremankit-<data>`.

### Przejście ze starej nazwy (agentkit)

Projekt nazywał się wcześniej `agentkit`. `python install.py` uruchomiony na maszynie ze starą instalacją robi migrację: `~/.config/agentkit` (klucze API, stan) jest przenoszony do `~/.config/foremankit`, stary blok reguł w `~/.claude/CLAUDE.md`, stare zadania autostartu VS Code i stara rejestracja MCP są usuwane, a nowe instalowane. Zmienne `AGENTKIT_*` nadal działają (są kopiowane do `FOREMANKIT_*`, gdy nowa nazwa nie jest ustawiona). Nadpisanie projektu `.agentkit/conventions.md` trzeba zmienić na `.foremankit/conventions.md`.

### Odinstalowanie i przywracanie

| Polecenie | Efekt |
|---|---|
| `python install.py --check` | tylko odczyt: lista śladów foremankita, które nadal są zainstalowane (kod wyjścia 1, gdy coś zostało) |
| `python install.py --uninstall` | cofa to, co dodał foremankit, i zostawia Twoje późniejsze zmiany (szczegóły niżej) |
| `python install.py --uninstall --stop-services` | dodatkowo zatrzymuje działającą bramkę i `llama-server` z tego folderu |
| `python install.py --uninstall --purge` | dodatkowo kasuje migawkę stanu sprzed instalacji oraz pliki robocze bramki |
| `python install.py --restore-original` | przywraca co do bajta pliki sprzed pierwszej instalacji (późniejsze zmiany w nich przepadają) |

`--uninstall` usuwa: agentów, blok reguł w `CLAUDE.md`, cztery hooki, rejestrację MCP oraz, jeśli użyto `--local-orchestrator`, wpisy bramki w `settings.json` VS Code, autostart (wpis Harmonogramu zadań i zadania w `tasks.json`) oraz `task.allowAutomaticTasks` (tylko jeśli ustawił je instalator). Agent, który istniał przed instalacją, wraca na miejsce; pliki utworzone przez instalator znikają, jeśli są puste. Na końcu wypisuje, co jeszcze zostało.

`--stop-services` jest bezpieczne: bramka jest zatrzymywana dopiero po wyczyszczeniu ustawień i tylko jeśli nic już na nią nie wskazuje (ustawienia VS Code, `env` w `~/.claude/settings.json`, zmienna `ANTHROPIC_BASE_URL` w powłoce). W przeciwnym razie wtyczka straciłaby połączenie także z chmurą, więc instalator odmawia i wypisuje, co nadal wskazuje na bramkę. Bez tej flagi bramka i serwer nadal działają.

`--restore-original` bierze pliki z migawki (`CLAUDE.md`, `settings.json`, pliki agentów i, jeśli instalowano orkiestrator, pliki VS Code). Przed nadpisaniem każdy plik dostaje kopię `*.bak-foremankit-*`. Migawkę robi pierwsze uruchomienie `install.py`; pliki VS Code dochodzą do niej, gdy po raz pierwszy użyjesz `--local-orchestrator`. Przy starszej instalacji bez migawki używana jest najstarsza kopia `*.bak-foremankit-*`; plik bez kopii zostaje nietknięty.

Zostaje celowo: venv, klucze API w `~/.config/foremankit/.env`, dziennik użycia i pliki `.agent-out/` w projektach (w tym `checkpoint-*.md`). Po odinstalowaniu **zrestartuj VS Code**, bo wtyczka czyta adres bramki przy starcie. Rejestrację MCP cofa tylko `--uninstall`.

Co się dzieje z modelem w wtyczce: wpis „Qwen (lokalny)” w pickerze to etykieta dla identyfikatora `claude-sonnet-5`, który bramka przekierowuje do lokalnego Qwena. Po odinstalowaniu picker traci tę pozycję, a identyfikator `claude-sonnet-5` trafia do chmury, czyli zapamiętany wybór oznacza prawdziwego Sonneta 5. Wybierz po restarcie docelowy model w pickerze.

**Przenoszenie na inną maszynę lub do innego projektu:** skopiuj folder `foremankit/` (bez `.venv`) i uruchom `python install.py`. W kodzie nie ma ścieżek bezwzględnych. Jedyna ścieżka bezwzględna to komenda MCP zapisywana przez `claude mcp add`, którą instalator generuje na bieżącej maszynie. Serwer MCP działa w katalogu bieżącego projektu, więc wyniki trafiają do `<projekt>/.agent-out/`. Ten folder sam dodaje swój `.gitignore`.

### Lokalny orkiestrator (wtyczka Claude Code → gateway → Qwen)

Opcjonalnie wtyczka Claude Code w VS Code może używać lokalnego Qwena (`llama-server`) jako modelu głównej rozmowy, a ten sam model dalej pracuje jako subagent/węzeł puli `llm`.

```bash
python install.py --local-orchestrator    # do ~/AppData/Roaming/Code/User (settings.json, tasks.json); cofa to --uninstall
```

- **Gateway** (`foremankit/gateway.py`, `foremankit-gateway`, port 8081): wtyczka łączy się z nim przez `ANTHROPIC_BASE_URL`. Żądania modelu z `local_models` (picker „Qwen (lokalny)" = `claude-sonnet-5`) idą do `llama-server` (nazwa modelu przepisana, dane logowania usunięte, dokładany `FOREMANKIT_PEER_KEY`). Wszystko inne idzie bez zmian do `api.anthropic.com`, więc modele chmurowe działają jak zwykle. **Gdy gateway nie działa, wtyczka nie dociera do żadnego modelu.**
- **Odchudzanie promptu** (`foremankit/gateway_trim.py`, ustawienia w `config/orchestrator.yaml`, bez restartu): Claude Code wysyła 66–100 tys. tokenów na żądanie, a lokalny model liczy je na CPU. Dla głównej pętli wtyczki gateway (1) zastępuje prompt systemowy treścią `config/orchestrator.md` plus sekcją `# Environment` z oryginału, (2) zostawia tylko narzędzia z `tools_keep` i skraca ich opisy do `tool_description_max_chars`, (3) usuwa bloki `<system-reminder>` pasujące do `drop_message_blocks` (lista skilli, instrukcje MCP, lista agentów). Główną pętlę rozpoznaje po narzędziu `Agent`; **subagenci i modele chmurowe nie są zmieniane**. Linia w logu pokazuje efekt: `(main loop: ~NNk -> ~MMk tokens)`. `FOREMANKIT_GATEWAY_DUMP=<plik>` zapisuje ostatnie przekształcone żądanie do strojenia listy narzędzi.
- **Serwer**: `llama/start-orchestrator.cmd` = `start-llama.ps1 -NoMenu -Share` (kontekst, offload, KV cache i myślenie z dostrojonego profilu w `llama/profiles.json`). Gateway zmniejsza żądania do ok. 25 tys. tokenów, więc 64 tys. wystarcza, a KV w `q8_0` zostaje na GPU. Wariant 196608 z KV w RAM (`--no-kv-offload`) dał 16 tok/s na pustym kontekście i 3–4 na długim (przy 32 tys. na GPU było 40). `-NCpuMoe` powyżej 34 załamuje prędkość. Myślenie jest włączone z limitem 1024 tokenów (`FOREMANKIT_THINK_BUDGET`, `0` = wyłączone): w teście nie spowolniło generowania, a odpowiedzi rzadziej wymagały dopytywania; bez limitu budżet tokenów znika w bloku `thinking`. Skrypt odmawia drugiej instancji.
- **Panel**: `python install.py --local-orchestrator` podlinkowuje `vscode-extension/` do VS Code (raz przeładuj okno). W pasku bocznym pojawia się przycisk ForemanKit: stan routera i llama-server (model, zapełnienie kontekstu), start i stop serwera z wyborem modelu, strojenie jednym kliknięciem (offload, wątki i największy szybki kontekst), przełącznik myślenia lokalnego modelu dla każdego poziomu wysiłku, start i stop routera w trybie bezpośrednim (niżej) oraz skróty do `profiles.json`, `orchestrator.yaml` i `router.log`. Logika jest w `foremankit/panel.py` (testowalna), a rozszerzenie to cienka nakładka bez etapu budowania.
- **Binarka llama-server**: nie ma jej w gicie. `python install.py --local-orchestrator --with-llama` (albo `python llama/get_llama.py`, albo przycisk w panelu widoczny, gdy jej brakuje) pobiera przetestowany build llama.cpp (`b11429`; `--latest` dla najnowszego) dobrany do maszyny: CUDA 13.4 lub 12.4 z runtime dla dość nowego sterownika NVIDIA, Vulkan dla innych GPU, w przeciwnym razie CPU. Każde archiwum jest sprawdzane względem sumy sha256 publikowanej przez GitHub. Pliki modeli (`.gguf`) nie są pobierane: wrzuć własne do `llama/`.
- **Podpowiedź o delegowaniu**: hook `delegation_nudge.py` (PostToolUse dla Write, Edit i Bash) liczy linie, które sesja pisze sama, także skrypty zapisywane przez heredoc w powłoce, od ostatniego wywołania `delegate` lub agenta. Po 150 liniach (`FOREMANKIT_NUDGE_LINES`) przypomina modelowi i pokazuje Tobie jednolinijkową notkę, że testy, dokumentacja, osobne pliki i mechaniczne zmiany w wielu plikach należą do tańszych modeli; powtarza się dopiero po kolejnych 150 liniach. Gdy delegowanie jest wyłączone, milczy, a część `UserPromptSubmit` mówi modelowi, żeby robił wszystko sam i nigdy nie wspominał o delegowaniu. Reguły, które instalator wpisuje do `~/.claude/CLAUDE.md`, zaczynają się sześciopunktową listą kontrolną; długa dokumentacja następuje po niej.
- **Przełącznik delegowania**: sekcja *Delegation* ma główne pole wyboru i po jednym dla kategorii źródeł (chmury płatne, chmury darmowe, modele lokalne); to istniejące przełączniki `foremankit delegation [paid|free|local|all] [on|off]`. Przy wszystkim wyłączonym narzędzie `delegate` odpowiada `DELEGATION IS SWITCHED OFF by the user ... Do this task yourself; do not retry delegation`, więc sesja Claude robi pracę sama i nic nie wydaje u innych dostawców. Ustawienie jest czytane przy każdym wywołaniu: bez restartu.
- **Podgląd na żywo i koszty**: pod lampkami stanu panel pokazuje, co lokalny model robi w tej chwili: *czyta prompt*, *myśli*, *odpowiada* albo *wywołuje narzędzie*, z liczbą tokenów na sekundę (gateway obserwuje strumień odpowiedzi, `foremankit/live.py`; mniej więcej jedna paczka na token, więc tempo jest przybliżone), a przy zadaniach delegowanych postęp czytania promptu z `/slots` serwera. Sekcja *Delegation* czyta log wywołań routera (`calls.jsonl`): koszt dzisiejszy i z ostatnich 7 dni, dzisiejsze wywołania według celu (dostawca, węzeł, model) oraz ostatnie delegowane zadania z rodzajem, tokenami, kosztem i czasem. Wywołania darmowe i lokalne kosztują 0.
- **Parowanie komputerów labu**: przyciski parowania w sekcji *Extra workers* zastępują ręczne kopiowanie klucza labu (`foremankit/pairing.py`). Na komputerze, który ma klucz, naciśnij *Show pairing code* (jeśli to pierwszy komputer, klucz zostanie utworzony): przez 2 minuty widać 8-znakowy kod, a komputer odpowiada na zapytania wykrywające (UDP 8766) i czeka na jednego dołączającego (TCP 8767). Na nowym komputerze naciśnij *Find computers*, wpisz kod i *Pair*. Klucz idzie zaszyfrowany AES-GCM kluczem wyprowadzonym z kodu (scrypt), host wydaje jeden zaszyfrowany klucz na sesję, a dołączający dowodzi, że go otworzył, więc podsłuch w sieci nic nie daje bez kodu. Potem oba komputery są zarejestrowane jako wzajemni parobkowie (dozwolone dane prywatne, priorytet 5), a klucz ląduje w `.env`; inny klucz, który już tam jest, jest zastępowany dopiero po Twoim potwierdzeniu. Rozwiązanie na sieć domową, gdzie każdy komputer jest Twój: wykrywanie przez broadcast bywa blokowane w sieciach firmowych, a Windows raz zapyta o zgodę dla Pythona w sieciach prywatnych. Żeby komputer był używany, jego llama-server nadal musi działać z „Share with the lab”.
- **Zapora do parowania (Windows)**: parowanie potrzebuje ruchu przychodzącego UDP 8766 i TCP 8767. Okienko Windows „zezwól na dostęp” domyślnie oferuje tylko sieci prywatne, a odrzucone okienko zostawia regułę *blokującą* ruch przychodzący dla `python.exe`, która przebija każdą regułę zezwalającą (przez nią Windows też przestaje pytać). Przycisk *Open firewall (Private + Public)* w *Extra workers* uruchamia z podniesionymi uprawnieniami `llama/pair-firewall.ps1` (jedna zgoda UAC): tworzy reguły zezwalające dla wyrzutni venv i dla bazowego interpretera, który naprawdę otwiera gniazdo, na profilach Private i Public, ale tylko dla lokalnej podsieci, i usuwa te pozostałe reguły blokujące dla tych samych programów. `llama/pair-firewall.ps1 -Remove` usuwa reguły; porty odpowiadają tylko, gdy trwa sesja parowania.
- **Zarządca na innym PC**: w panelu „Orchestrator for Claude Code runs on” wymienia ten komputer i każdego parobka z puli. Wybór parobka kieruje lokalne żądania gatewaya do llama-server na tej maszynie (`orchestrator_target.json` w katalogu stanu, bez restartu). Ta maszyna musi mieć serwer uruchomiony z `-Share` (robi to `llama/start-orchestrator.cmd`) i ten sam `FOREMANKIT_PEER_KEY`; pokazywane okno kontekstu pochodzi z tego serwera. Każdy parobek ma też pole wyboru, czy router może mu wysyłać zadania, a dodatkowych parobków można edytować i usuwać (parobki z `nodes.yaml` edytuje się w pliku).
- **Role i parobkowie**: lokalny serwer bywa *zarządcą* (odpowiada Claude Code przez gateway, bez zadań delegowanych), *parobkiem* (tylko zadania delegowane, Claude Code dostaje czytelne 503; to jest domyślne) albo *obydwoma*; panel to przełącza (plik roli gatewaya i przełącznik routera `llm:self`). Dodatkowych parobków llama-server (inne maszyny lub porty) dodaje się w panelu, zapis w `~/.config/foremankit/workers.json` i scalanie z pulą węzłów obok `nodes.yaml`; panel pokazuje ich status. Formularz „Advanced parameters” zapisuje ręcznie dobrane parametry uruchomienia (kontekst, eksperty na CPU, wątki, KV cache, myślenie, temperaturę) w profilu modelu, jako alternatywa dla autostrojenia.
- **Router a połączenie z Anthropic**: Claude Code łączy się z Anthropic przez router (`ANTHROPIC_BASE_URL` w ustawieniach VS Code), więc zatrzymany router oznacza brak połączenia. W trybie routera wpis w Harmonogramie zadań podnosi martwy router w ciągu 5 minut; **tryb bezpośredni** („Stop (direct mode)” w panelu), który zatrzymuje router i usuwa go z ustawień VS Code, więc Claude Code łączy się wprost z Anthropic. Claude Code czyta to ustawienie tylko przy starcie okna, więc **po każdym przełączeniu trzeba przeładować okno VS Code** (panel najpierw pyta, potem sam przeładowuje okno).
- **Tryb bezpośredni jest domyślny**: `python install.py --local-orchestrator` instaluje tylko panel (link w folderze rozszerzeń VS Code). Claude Code nadal łączy się wprost z Anthropic; nie ma routera, wpisu w Harmonogramie zadań, wpisów gatewaya w ustawieniach VS Code ani zadań w `tasks.json`. **Tryb routera** włącza się świadomie: `--router` albo „Start router” w panelu. Zapisuje trzy wpisy gatewaya w ustawieniach VS Code, rejestruje w Harmonogramie wpis `ForemanKit router` (przy logowaniu i co 5 minut, więc martwy router wraca; poza Windows zamiast tego zadanie VS Code przy otwarciu folderu) i uruchamia gateway. „Stop (direct mode)” w panelu albo `--uninstall` usuwa wszystkie te ślady i zatrzymuje router. llama-server nigdy nie startuje automatycznie.
outer.log`.
- Znane ograniczenia: serwer bez `mmproj` nie przyjmuje obrazów (zrzut ekranu w historii daje HTTP 500, wtedy nowa rozmowa); pierwsza odpowiedź w sesji przetwarza cały prompt. Cofnięcie: `python install.py --uninstall`.

### Klucze

Klucze ustawiasz wyłącznie w zmiennych środowiskowych albo w pliku `.env`: `foremankit/.env` lub `~/.config/foremankit/.env`. Wzór jest w `.env.example`. Klucze nigdy nie trafiają do configów, logów ani do `~/.claude.json`, bo rejestracja MCP celowo **nie** używa `-e KEY=...`.

Provider jest aktywny tylko wtedy, gdy jego klucz jest ustawiony.

| Provider | Zmienna | Typ | Uwagi |
|---|---|---|---|
| deepseek | `DEEPSEEK_API_KEY` | **płatny** | flash domyślny, pro przy eskalacji; liczy się koszt z `usage` (cache hit/miss). Modele: `deepseek-flash` (V4.1 Flash) i `deepseek-v4-pro`; ceny za 1M tokenów w szczycie (poza szczytem połowa): flash 0,30 wejście / 1,20 wyjście, pro 1,32 / 3,96. Rozumowanie jest domyślnie włączone i liczone jak wyjście, więc router steruje nim poziomem z `routing.yaml`: `minimal` i `low` wyłączają je, `medium` włącza z niskim wysiłkiem, `high` z wysokim. Przy `medium` z wysiłkiem `high` zmierzyłem 3,5 tys. tokenów rozumowania na małe zadanie i brak odpowiedzi, stąd niski wysiłek. `enabled: auto`: foremankit sprawdza darmowym `GET /models` (wynik w cache 10 min), czy DeepSeek odpowiada, i jeśli tak, **wola go od płatnego Gemini**. Filtr sieci firmowej blokuje `api.deepseek.com`, więc w pracy jest po cichu pomijany, a w domu włączy się sam. Nie obchodź blokady. `foremankit status` pokazuje wynik testu |
| gemini_free | `FREE_GEMINI_API_KEY` | darmowy (drugie konto Google) | osobny limit od płatnego klucza, używany razem z innymi darmowymi przed płatnymi. `trains_on_data: true`. **Domyślnie wyłączony** (pomiar 2026-10-08: 50 % wywołań kończyło się błędem, razem z OpenRouterem obsłużył grosze) |
| gemini | `GEMINI_API_KEY` | **płatny** (konto z włączonymi płatnościami) | `flash` do zwykłej pracy, `pro` zastępuje DeepSeek Pro w `hard_code`. Każde wywołanie liczy się do budżetu USD (zakładamy najgorszy przypadek). Ceny w configu obowiązują do 31.12.2026, od 1.01.2027 Flash drożeje dwukrotnie. Nigdy dla danych prywatnych |
| groq | `GROQ_API_KEY` | darmowy | szybki, wąski limit tokenów/dzień |
| openrouter | `OPENROUTER_API_KEY` | darmowy (`:free`), **domyślnie wyłączony** | ok. 20 rpm, 50 rpd (1000 rpd po wpłacie 10 USD) |
| mistral | `MISTRAL_API_KEY` | darmowy (Experiment) | |
| cerebras | `CEREBRAS_API_KEY` | darmowy | szybki (gpt-oss-120b); limity darmowego planu do potwierdzenia |
| llm | `LLAMACPP_URL` (główny, domyślnie `http://127.0.0.1:8080`), `OLLAMA_LOCAL_URL` (opcjonalny zapas, domyślnie `http://localhost:11434`) | lokalny | Opcjonalnie `LOCAL_LLM_GEN_TPS` i `LOCAL_LLM_PROMPT_TPS` nadpisują `speed` z `providers.yaml` na danej maszynie (router szacuje po nich czas zadania); wartości weź z `tg_tok_s`/`pp_tok_s` w `llama/profiles.json` |

---

## Komendy

```bash
foremankit status                     # providerzy, pozostałe limity, wydatki dziś/miesiąc, węzły puli LLM
foremankit models [provider]          # lista z /models każdego aktywnego providera; oznacza ID z configu, których nie ma
foremankit stats [--days 7]           # koszt i wywołania per provider/kind + przybliżona oszczędność vs ceny API Sonneta
foremankit nodes                       # status wszystkich węzłów: osiągalność, zajętość, prędkość, reguła, akceptowany poziom danych, netguard
foremankit nodes key                   # tworzy FOREMANKIT_PEER_KEY w .env
foremankit llm status                  # wszystkie węzły
foremankit llm test                     # test wszystkich węzłów
foremankit llm <node> test              # test konkretnego węzła
foremankit llm <node|self|others|all> on|off|auto
foremankit profile [nazwa]            # pokazuje albo przełącza profil (home|work|...)
foremankit delegation [paid|free|local|all] on|off|status   # przełącza całą kategorię źródeł
foremankit check plik.py [...]       # sprawdza pliki Pythona z regułami czystego kodu (config/conventions.md)
foremankit review [--diff] pliki     # recenzja przez drugi model (własny LLM albo płatny); --diff recenzuje git diff
python install.py --check          # co z foremankita jest zainstalowane (osobno od CLI, w folderze foremankita)
foremankit delegate "zadanie" --kind quick [--sensitivity private] [--files a.py b.py] [--escalate] [--allow-cost] [--verify "pytest"] [--verify-target "test_a.py"]
```

Na Windows bez aktywacji venv: `.venv\Scripts\foremankit.exe status`.

`FOREMANKIT_DRY_RUN=1` zwraca atrapę odpowiedzi bez żadnego wywołania API. Służy do sprawdzenia, który provider zostałby wybrany.

---

## Jak to działa

### Narzędzia MCP (tylko dwa, krótkie opisy, ok. 250 tokenów na sesję)

- `delegate(task, kind, sensitivity="normal", context_files=[], system_hint="", max_output_tokens=4000, escalate=false, allow_cost=false, verify="", verify_target="")` zapisuje pełną odpowiedź do `.agent-out/<id>.md`. Do Claude wraca tylko provider, model, tokeny, koszt, ścieżka i podgląd (ok. 600 znaków), **nigdy cała odpowiedź**.
- `router_status()` pokazuje aktywnych providerów, limity, wydatki i węzły puli LLM.

### Routing (`config/routing.yaml`)

| kind | łańcuch (pierwszy dostępny wygrywa) |
|---|---|
| bulk_code – boilerplate, testy, docs, refaktor | mistral → groq → llm → deepseek:flash → gemini:flash (lokalny model tylko dla małych wejść, `llm.kinds`; pomiar w `benchmarks/CLOUD_VS_LOCAL.md`) |
| data_script – pandas/SQL/czyszczenie | deepseek:flash → gemini:flash |
| hard_code – złożony kod | deepseek:pro → gemini:pro |
| long_context – duże logi, wiele plików | deepseek:flash → gemini:flash |
| quick / summarize | groq → cerebras → llm → gemini:flash |

| private (lub `sensitivity="private"`) | llm, **i nic więcej** (tylko maszyny z `accept: private`) |

Kolejność to: darmowi → DeepSeek (preferowany płatny) → płatny Gemini jako ostatnia rezerwa. Niedostępny lub zablokowany DeepSeek jest pomijany po cichu. Błąd połączenia (np. blokada) daje 30 s cooldownu, żeby kolejne zadania nie czekały na timeout.

`escalate=true` zamienia łańcuch na `escalation` (deepseek:pro). Nazwy modeli są tylko w `providers.yaml`.

Router pomija providera, gdy:
- nie ma klucza albo provider jest wyłączony;
- wyczerpał się limit rpm/rpd/tpd (liczniki w `~/.config/foremankit/state.json`, doba wg UTC);
- trwa cooldown po 429 (backoff 30 s → 2 min → 10 min → 30 min albo `Retry-After`) lub po 5xx;
- dla płatnych: najgorszy możliwy koszt wywołania (szacowane wejście + `max_output_tokens`) przekroczyłby budżet. To **twardy stop**: obecnie 1 USD/dzień i 5 USD/miesiąc (tyle, ile wpłacono na Gemini), ustawiany w `routing.yaml → budget`.

Ponowienia: maksymalnie 1 i tylko po błędzie 5xx lub połączenia. Po timeoucie (zawieszony serwer) i po 429 router od razu przechodzi do następnego providera.

**Co idzie na wolny lokalny model.** Jawna lista w `routing.yaml → llm.kinds`: obecnie `quick` (do 6 tys. tokenów wejścia), `summarize` (8 tys.), `bulk_code` i `data_script` (po 5 tys.). Rodzaje spoza listy (`hard_code`, `long_context`) i wejścia powyżej limitu **od razu** idą do następnego dozwolonego źródła (płatny Gemini, a po włączeniu `FOREMANKIT_ALLOW_INTERNAL` także DeepSeek), żeby długie zadanie nie blokowało jedynego slotu. Dodatkowo działa szacunek czasu z prędkości węzła (`nodes.yaml → speed`, limit `max_seconds`). Jedyny wyjątek to dane `private`: nie mają innego dozwolonego źródła, więc lokalny model dostaje je niezależnie od rozmiaru. Lokalnie rozumowanie jest wyłączone dla poziomów poniżej `high` (zmierzone: 23 s zamiast 142 s i pustej odpowiedzi).

**Równoległość.** Kilka wywołań `delegate` naraz (np. w jednej wiadomości Claude'a) działa równolegle i router rozdziela je między źródła. Lokalny serwer obsługuje jedno zapytanie naraz (`max_concurrent: 1`), więc drugie równoległe zadanie nie czeka w kolejce, tylko przechodzi do następnego dozwolonego źródła. Jeśli to jedyne dozwolone źródło (np. `private`), router poczeka na jego wolny slot zamiast zgłosić błąd. Dla danych `internal` lokalny model jest próbowany **pierwszy** (darmowy i prywatny), poza `hard_code` i `escalate=true`, gdzie liczy się najmocniejszy model.

### Ochrona kosztów

- `budget.per_call_usd = 0.10`: wywołanie płatnego modelu, którego najgorszy możliwy koszt przekracza 0.10 USD, jest odrzucane, chyba że wywołanie przekazuje `allow_cost=true` (CLI `--allow-cost`; zapytaj użytkownika najpierw).
- `reasoning.large_output_tokens 6000 / large_output_level medium`: żądanie odpowiedzi powyżej 6000 tokenów obniża poziom rozumowania do `medium`.
- `calls.warn_output_tokens 8000`: żądanie odpowiedzi powyżej 8000 tokenów daje notatkę „podziel zadanie”.
- `calls.abort_waste_usd 0.02`: wywołanie płatnego modelu, które po zużyciu ponad 0.02 USD zwraca pustą odpowiedź lub obciętą wewnątrz bloku kodu, kończy się błędem `TaskTooLarge` i nie jest ponawiane na innym modelu (częściowy tekst pozostaje w `.agent-out`). Ta zasada chroni przed kosztownymi błędami.

### Weryfikacja wyniku (`verify`)

Dla kodu, który da się sprawdzić komendą, podaj `verify` i `verify_target` (CLI: `--verify`, `--verify-target`):

```bash
foremankit delegate "Napisz testy dla verify.py" --kind bulk_code --files foremankit/verify.py \
  --verify ".venv/Scripts/python -m pytest tests/test_verify_unit.py -q" --verify-target tests/test_verify_unit.py
```

- Router zapisuje blok kodu z odpowiedzi do `verify_target` i uruchamia komendę (bez powłoki, w folderze projektu, limit `verify.timeout_s` = 300 s).
- Gdy komenda nie przechodzi, jej wyjście wraca do **tego samego** modelu z prośbą o poprawiony plik (maks. `verify.max_rounds` = 2 razy).
- Plik zostaje zmieniony **tylko**, gdy komenda przejdzie. W przeciwnym razie wraca oryginał (nowy plik jest usuwany). Wynik pokazuje `verify: PASSED ...` albo `verify: FAILED ... was restored`.
- Bezpieczeństwo: komenda musi zaczynać się od prefiksu z `routing.yaml → verify.allowed_commands` (`python -m pytest`, `.venv/Scripts/python -m pytest`, `.venv/bin/python -m pytest`, `pytest`, `python -m ruff check`, `ruff check`). `verify_target` to zwykły plik względny wewnątrz projektu (nigdy `.env`, `.git`, `.ssh`). Zły wniosek jest odrzucany, zanim zapłacisz za model, a komenda, która się nie uruchamia, kończy pętlę od razu.
- Ścieżki liczą się od katalogu roboczego procesu: serwer MCP pracuje w folderze sesji Claude, a CLI w folderze, z którego go uruchomisz.

### Jeden standard kodu (clean code)

Cały kod pisany przez foremankit i dla niego ma trzymać te same zasady, niezależnie od narzędzia i sesji: czysty kod wg R. Martina, czytelny dla początkującego programisty (nazwy ujawniające intencję, małe funkcje, brak duplikacji i magicznych wartości, konkretne wyjątki, logika oddzielona od I/O, docstring przy każdym module, klasie i funkcji publicznej, komentarze wyjaśniające **dlaczego**). Pełny tekst to `config/conventions.md`.

- **Prompty:** dla rodzajów `bulk_code`, `data_script` i `hard_code` router dokleja ten plik do promptu systemowego (ok. 600 tokenów na wywołanie; prompt jest stały, więc cache DeepSeeka działa). Projekt może go nadpisać plikiem `<projekt>/.foremankit/conventions.md`.
- **Sesje Claude:** skrót standardu jest w regułach `claude/CLAUDE.global.md` (instaluje je `install.py`), więc ta sama konwencja obowiązuje w każdej sesji.
- **Sprawdzanie maszynowe:** `foremankit check plik.py` pilnuje tego, co da się zmierzyć: długość funkcji (20 linii), liczba parametrów (4), zagnieżdżenie (3), brak docstringów, goły `except`, jednoliterowe nazwy. Limity są w `routing.yaml → conventions.limits`. Użyj w `verify`, a plik trafi do repo dopiero po spełnieniu zasad: `--verify ".venv/Scripts/foremankit check plik.py" --verify-target plik.py`. Czy nazwa jest *dobra* albo komentarz *wyjaśnia dlaczego*, ocenia nadal prompt i człowiek.
- Cały kod w `foremankit/` spełnia te limity (`foremankit check foremankit/*.py` mówi: clean). Jedyny świadomy wyjątek to `server.delegate`: jego 11 parametrów to pola schematu narzędzia MCP; oznacza je komentarz `# codecheck: allow too-many-parameters (powód)`. Tylko ta reguła da się wyciszyć, a wyciszenie bez powodu w nawiasach jest ignorowane.

### Wstępne code review przez drugi model

`delegate(..., review=true)` (CLI `--review`) uruchamia po `verify` recenzję **innego** modelu niż autor: własnego `llm` (pula węzłów, najpierw szybki Legion) albo płatnego (`deepseek:flash`, `gemini:flash`). Darmowe chmury nigdy nie recenzują (płytko, a część trenuje na kodzie), a dane `private` zostają na własnych węzłach. Łańcuch to `chains.review` w `routing.yaml`; wolny węzeł dostaje recenzję na końcu (`llm.kinds.review`, `max_seconds`).

- Recenzent odpowiada jedną linią na problem: `SEVERITY | gdzie | problem | dowód | poprawka`, gdzie dowód to dosłowny cytat kodu. Uwaga jest **uzasadniona** tylko wtedy, gdy cytat naprawdę występuje w kodzie; wymyślone uwagi są w raporcie oznaczone jako niezweryfikowane i nigdy nie uruchamiają poprawki.
- Uzasadnione uwagi o stopniu z `review.fix_severities` (domyślnie tylko `blocker`) wracają do autora **raz** (`review.fix_rounds`), a potem `verify` sprawdza plik ponownie. Jeśli poprawka psuje test, zostaje wersja sprzed niej. Reszta uwag trafia tylko do raportu.
- Brak recenzenta (nikt nie jest dostępny albo dane są zbyt poufne) nie przerywa zadania: raport ma wtedy `review: SKIPPED (powód)`.
- `foremankit review plik.py ...` albo `foremankit review --diff ścieżki` recenzuje istniejący kod (np. własne zmiany przed przekazaniem dalej); kod wyjścia 1, gdy jest uzasadniony `blocker`. Opcje: `--task "co ten kod ma robić"`, `--sensitivity`, `--allow-cost`.
- Koszt: wyjście do 1500 tokenów (`review.max_output_tokens`); lokalny węzeł jest darmowy, a płatny flash kosztuje ułamek centa za plik.

### Oszczędność

- Każdy `kind` ma stały prompt systemowy na początku, a zmienne dane (pliki, notatki, zadanie) idą na końcu. Dzięki temu DeepSeek trafia w cache prefiksu, który kosztuje ok. 10% ceny wejścia.
- `max_output_tokens` jest ograniczony do 16000, a każdy provider ma timeout.
- Log `~/.config/foremankit/calls.jsonl` zawiera tylko metadane: provider, model, tokeny, koszt, czas, sukces. **Treści promptów nie są logowane** (`log_prompts: false`).

---

## Prywatność (fail closed)

Trzy poziomy `sensitivity`:

| Poziom | Dla jakich danych | Kto może je dostać |
|---|---|---|
| `normal` | kod, dane publiczne lub zanonimizowane | wszyscy providerzy; do modeli zewnętrznych tylko schemat i kilka wierszy |
| `internal` | dane procesowe (tagi, wyciągi PI/SQL, dokumenty wewnętrzne) | tylko providerzy, którzy **nie trenują na danych** i mają jawne `allow_internal: true` (płatny Gemini), albo własne węzły puli LLM. Nigdy darmowe tiery. DeepSeek ma `trains_on_data: true` (jego polityka, sprawdzona 2026-10-08, dopuszcza użycie promptów do trenowania, a dane są przechowywane w Chinach), więc domyślnie jest wykluczony. Można go świadomie dopuścić **na danej maszynie** zmienną `FOREMANKIT_ALLOW_INTERNAL_TRAINING=deepseek` w `.env` (decyzja użytkownika z 2026-10-08: dane procesowe nie są tajemnicą). Dla `private` nie działa nigdy. Sieć firmowa blokuje DeepSeeka, więc zgodność z polityką firmy warto potwierdzić Limit linii dla plików danych jest zniesiony |
| `private` | hasła, dane osobowe, receptury i tajemnice handlowe | wyłącznie maszyny z `accept: private` |

Dla `internal` nie ma kontroli adresu sieciowego (Gemini to publiczny endpoint). Jeśli żaden uprawniony provider nie działa, dostajesz błąd i nic nie jest wysyłane gdzie indziej. Klasyfikację danych procesowych jako „wewnętrzne” rozstrzyga polityka Twojej firmy, więc w razie wątpliwości zapytaj IT.

- `sensitivity="private"` zawsze idzie do maszyn z `accept: private`, i tylko do nich. Gdy żadna nie działa, dostajesz **jasny błąd** i nic nie jest wysyłane do chmury. Router odrzuca chmurę nawet wtedy, gdy ktoś dopisze ją do łańcucha `private`.
- Providery z `trains_on_data: true` oraz DeepSeek nigdy nie dostają danych prywatnych.
- **netguard**: dane prywatne idą tylko na adresy localhost/127.0.0.0/8, 10.0.0.0/8, 172.16.0.0/12 i 192.168.0.0/16. Nazwa hosta jest rozwiązywana przez DNS i jeśli którykolwiek adres jest publiczny, router odmawia. Gdy DNS zawiedzie, też odmawia. Obejście to `allow_public_endpoint: true` przy danej instancji, z ostrzeżeniem w logu. Nie rób tego bez wyraźnego powodu.
- Strażnik plików: do modelu zewnętrznego plik danych (`.csv`, `.tsv`, `.jsonl` itp.) powyżej 60 linii jest odrzucany z prośbą o schemat i kilka zanonimizowanych wierszy. Pliki binarne (parquet, pickle) są odrzucane zawsze.
- Dane firmowe i procesowe (tagi PI, wyciągi SQL, receptury) traktuj jako `private`. Ta zasada jest też w regułach dla Claude.

---

## Pula węzłów LLM (`config/nodes.yaml`)

Lokalne modele (`llama-server` z llama.cpp albo Ollama; wystarczy serwer odpowiadający na `/v1/models` lub `/api/tags`) tworzą **pulę `llm`**. Każda maszyna labu to jeden węzeł, także ta, na której pracujesz. Ten sam `nodes.yaml` leży na każdej maszynie, a maszyna rozpoznaje siebie po nazwie hosta (albo po `FOREMANKIT_SELF`). Maszyna spoza pliku staje się węzłem `local` i korzysta z pozostałych. Pula jest przewidziana na sieć do ok. 5 komputerów. Maszyna bez GPU albo bez serwera ma `serves: false` (tylko korzysta z puli); serwer na samym CPU jest zwykłym węzłem z niską prędkością.

### Jak router wybiera węzeł

Gdy łańcuch dociera do wpisu `llm`, router:

1. pomija węzły wyłączone, nieosiągalne (wszystkie są sondowane równolegle, wynik pamiętany 30 s) i takie, które nie przyjmują poziomu danych zadania (`accept`);
2. dla pozostałych szacuje czas ukończenia: `tokeny_wejścia / prompt_tps + min(max_output, expected_output_cap) / gen_tps`, a dla węzła, którego serwer jest teraz zajęty (`GET /slots`), dodaje `llm.queue_penalty_s` (30 s). Prędkość to zapisane `speed` przesuwane ku temu, co węzeł **naprawdę dostarczył** (patrz niżej);
3. próbuje węzłów od najszybszego (remis wygrywa maszyna o wyższym priorytecie). Dzięki temu szybsza maszyna jest używana pierwsza **na każdej maszynie**, nie tylko na jednej;
4. węzeł, który jest za wolny dla zadania albo któremu rodzaj lub rozmiar zadania nie przysługuje (`routing.yaml → llm.kinds`, nadpisywane polem `kinds` węzła), dostaje zadanie dopiero w ostatniej kolejności. Dane `private` nie mają innego źródła, więc tam rozmiar nie blokuje.

Limity, kwoty i cooldown oraz statystyki są liczone **osobno dla każdego węzła** (`llm@legion`), więc awaria jednej maszyny nie wyłącza pozostałych.

### Pola węzła w `nodes.yaml`

| Pole | Znaczenie |
|---|---|
| `hostname` | nazwa maszyny, po której plik rozpoznaje „mnie” |
| `lan` | adresy sprawdzane po kolei (nazwa, IP albo pełny URL `https://...`, który jest użyty bez zmian) |
| `port` | port serwera, domyślnie 8080 |
| `serves` | `false` = maszyna tylko korzysta z puli |
| `accept` | najwyższy poziom danych, jaki węzeł może dostać: `normal` < `internal` < `private`. **Domyślnie `private`**: maszyny labu są właśnie po to, żeby dane poufne nie wychodziły poza własny sprzęt. Obniż dla maszyny, której nie ufasz |
| `speed` | `prompt_tps`, `gen_tps`, `expected_output_cap`, `max_seconds`: zmierzona prędkość (`llama/tune_llama.py`, wyniki w `llama/profiles.json`) |
| `learn` | `true` (domyślnie): router uczy się prędkości węzła z prawdziwych wywołań; `false`: zapisane `speed` jest ostateczne (dla węzłów zmierzonych dokładnie) |
| `kinds` | opcjonalnie: limity wejścia per rodzaj zadania dla tego węzła |
| `max_concurrent` | ile zapytań naraz przyjmuje serwer (domyślnie 1) |
| `priority` | mniejsza liczba = ważniejsza maszyna (służbowa przed prywatną); rozstrzyga remisy i reguły pożyczania |

Sekcje `settings` (`key_env`, `loan_to_higher_priority`) i `defaults` (wartość pola dla wszystkich węzłów) są opisane w komentarzach pliku.

### Uczenie się prędkości

Zapisane `speed` to oczekiwanie. Prędkość cudzej maszyny (laptop kolegi, wspólny serwer) zależy od jej obciążenia i sieci i **pływa**, więc router po każdym udanym wywołaniu węzła z `learn: true` aktualizuje średnią kroczącą tego, co węzeł dostarczył: dokładnie z pola `timings` odpowiedzi `llama-server`, a gdy go nie ma, szacunkowo z czasu odpowiedzi (przy odpowiedziach od 64 tokenów). Wartość nauczona przejmuje planowanie stopniowo: przy `n` pomiarach ma wagę `n / min_samples` (domyślnie 3), a węzeł nieużywany dłużej niż `max_age_h` (72 h) wraca do zapisanej prędkości. Parametry są w `nodes.yaml → settings.speed_learning`. Własne, dokładnie zmierzone laptopy mają `learn: false`. `foremankit nodes` pokazuje, czy prędkość jest przypięta, nauczona czy jeszcze tylko zapisana.

### Pożyczanie od ważniejszej maszyny

Maszyna pożyczająca od maszyny o **wyższym priorytecie** robi to tylko wtedy, gdy jej serwer jest wolny (`/slots`), i tylko dla zadań szacowanych na krócej niż `loan_to_higher_priority.max_seconds` (45 s). Właściciel nigdy nie czeka długo za cudzym zadaniem. Gdy wszystkie maszyny mają ten sam priorytet, ograniczeń nie ma.

### Klucz i bezpieczeństwo sieci

Każdy serwer węzła działa z `--api-key`, a wspólny klucz to `FOREMANKIT_PEER_KEY` w `.env` (nie w gicie). Utworzysz go poleceniem `foremankit nodes key` i musisz ręcznie skopiować tę samą linię do `.env` każdej maszyny. Dane `private` idą tylko na adresy przechodzące netguard (opisany wyżej) i tylko do węzłów z `accept: private`.

### Przełączniki i diagnostyka

```bash
foremankit nodes                        # tabela: osiągalny, zajęty, prędkość, reguła, przyjmowany poziom danych, netguard
foremankit nodes key                    # tworzy FOREMANKIT_PEER_KEY w .env
foremankit llm                          # stan wszystkich węzłów (modele, wybrany model)
foremankit llm test                     # sonda + krótkie generowanie na każdym węźle (albo: foremankit llm legion test)
foremankit llm others off               # wyłącz inne maszyny; też: self, all, albo nazwa węzła; on | off | auto
```

Zmienne środowiskowe (priorytet: stan z CLI > env > config; wygrywa najbardziej szczegółowe: węzeł, potem `self`/`others`, potem `all`):

- `FOREMANKIT_NODE_<NAZWA>_ENABLED`, `LLM_SELF_ENABLED`, `LLM_OTHERS_ENABLED`, `LLM_ENABLED` (`auto|on|off`);
- `FOREMANKIT_NODE_<NAZWA>_LAN="ip1,ip2"` : adres bez edycji pliku, np. przy zmiennym DHCP;
- `LLAMACPP_URL` (domyślnie `http://127.0.0.1:8080`) i `OLLAMA_LOCAL_URL` (domyślnie `http://localhost:11434`): serwery na tej maszynie, sprawdzane w tej kolejności;
- `LOCAL_LLM_GEN_TPS`, `LOCAL_LLM_PROMPT_TPS`: nadpisują prędkość **tylko tej maszyny**;
- `FOREMANKIT_SELF` (inna nazwa maszyny niż hostname), `FOREMANKIT_NODES=off` (zignoruj `nodes.yaml`).

Profile (`config/profiles.yaml`) używają kluczy `"llm:self"`, `"llm:others"`, `"llm:all"` lub `"llm:<węzeł>"`.

> **Zmiana nazw.** Providery `llm_local` / `llm_remote` (wcześniej `ollama_*`), plik `peers.yaml`, polecenia `foremankit ollama` i `foremankit peers` oraz zmienne `LLM_LOCAL_ENABLED`, `LLM_REMOTE_ENABLED`, `LLM_REMOTE_*_URL`, `OLLAMA_REMOTE_*` zostały usunięte bez aliasów. Stary config jest odrzucany z błędem i podpowiedzią (`peers.yaml` → `nodes.yaml`, klucz `peers:` → `nodes:`).

### Dostęp do węzła (LAN)

`foremankit llm test` sprawdza każdy węzeł: czas odpowiedzi, listę modeli, werdykt netguarda i krótką generację. Poniższe polecenia dotyczą Ollamy (port 11434). Dla `llama-server` użyj `--host <adres LAN> --api-key <klucz>` (launcher `llama/start-llama.ps1 -Share` i reguła zapory `llama/share-firewall.ps1`), port 8080.

**1. LAN (dom).** Na maszynie zdalnej:
```bash
# Linux (systemd): sudo systemctl edit ollama  ->  [Service] Environment="OLLAMA_HOST=0.0.0.0"
sudo ufw allow from 192.168.1.0/24 to any port 11434 proto tcp      # tylko podsieć domowa
```
```powershell
# Windows: zmienna systemowa OLLAMA_HOST=0.0.0.0, restart Ollamy, potem:
New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound -Protocol TCP -LocalPort 11434 -RemoteAddress 192.168.1.0/24 -Action Allow
```
Na laptopie ustaw `FOREMANKIT_NODE_<NAME>_LAN=http://192.168.1.50:11434`, gdzie 192.168.1.50 to adres maszyny zdalnej. Bezpieczniej jest zamiast `0.0.0.0` podać sam adres LAN, np. `OLLAMA_HOST=192.168.1.50`.

**2. Nigdy nie wystawiaj portu 11434 do internetu.** Ollama nie ma uwierzytelniania ani TLS.

---

## Konfiguracja i pliki

| Co | Gdzie |
|---|---|
| providerzy, modele, ceny, limity | `config/providers.yaml` |
| łańcuchy, prompty systemowe, budżet, strażnik plików | `config/routing.yaml` |
| profile | `config/profiles.yaml` |
| węzły | `config/nodes.yaml` |
| nadpisanie configu na danej maszynie | `~/.config/foremankit/config/*.yaml` lub `FOREMANKIT_CONFIG_DIR` |
| stan (liczniki, przełączniki, wydatki, cooldowny) | `~/.config/foremankit/state.json` (`FOREMANKIT_STATE_DIR`) |
| log wywołań | `~/.config/foremankit/calls.jsonl` |
| wyniki | `<projekt>/.agent-out/` (`FOREMANKIT_OUT_DIR`) |
| checkpointy hooka auto-checkpoint | `<projekt>/.agent-out/checkpoint-<czas>.md` |
| migawka stanu sprzed instalacji | `~/.config/foremankit/install-state/` (używa jej `install.py --restore-original`) |
| licznik kontekstu bramki | `~/.config/foremankit/gateway_context.json` |
| logi bramki (Windows) | `%TEMP%/router.log`, `%TEMP%/router.err` |
| ustawienia VS Code zmieniane przez `--local-orchestrator` | `settings.json` i `tasks.json` w folderze użytkownika VS Code (Windows: `%APPDATA%/Code/User`, Linux: `~/.config/Code/User`, macOS: `~/Library/Application Support/Code/User`) |
| kopie zapasowe przed zmianą pliku | obok pliku, jako `*.bak-foremankit-<data>` |

Pliki dla Claude (`claude/agents/*.md`, `claude/CLAUDE.global.md`) są po angielsku, bo są ładowane w każdej sesji i tak zużywają mniej tokenów.

### Pozycje VERIFY

Wszystko oznaczone `# VERIFY` w `providers.yaml` (ID modeli, limity, ceny DeepSeek, `trains_on_data`) oraz cena referencyjna Sonneta w `routing.yaml` są **niezweryfikowane**. Po ustawieniu kluczy uruchom `foremankit models`: wypisze ID, których brakuje na endpoincie. Ceny DeepSeek w configu są celowo zawyżone, żeby budżet zadziałał prędzej niż za późno.

## Rozwiązywanie problemów

| Objaw | Przyczyna i co zrobić |
|---|---|
| Wtyczka Claude Code w VS Code nie łączy się z żadnym modelem (także chmurowym) | Ustawienia VS Code wskazują na bramkę (`ANTHROPIC_BASE_URL` = `http://127.0.0.1:8081`), a bramka nie działa. Uruchom ją: `llama\Start-Router.ps1` (sprawdzenie: `-Status`). Jeśli bramki nie chcesz, usuń wpisy: `python install.py --uninstall`, potem zrestartuj VS Code. |
| Po odinstalowaniu wtyczka używa prawdziwego Sonneta 5 zamiast Qwena | Tak ma być: bez bramki identyfikator `claude-sonnet-5` trafia do chmury. Wybierz w pickerze docelowy model. |
| Narzędzie `delegate` nie jest widoczne albo zwraca błąd | Sprawdź rejestrację: `claude mcp get foremankit` oraz `python install.py --check`. Narzędzia MCP bywają „odroczone” (widoczne tylko z nazwy): załaduj ich schematy przez ToolSearch, zanim je wywołasz. |
| `install.py` pisze, że `settings.json` nie jest czystym JSON-em | Plik ma komentarze, więc instalator go nie rusza. Popraw ustawienia ręcznie albo usuń z niego komentarze i uruchom ponownie. |
| Nie wiem, co z foremankita jest zainstalowane | `python install.py --check` (tylko odczyt). |
| Chcę wrócić do stanu sprzed instalacji co do bajta | `python install.py --restore-original`, potem `--uninstall`, żeby wyrejestrować MCP. |
| Działają dwa serwery lokalnego modelu | Uruchamiaj serwer tylko przez `llama/start-llama.cmd`. `-List` pokazuje, co działa, `-Replace` podmienia, `-Force` służy do celowego drugiego serwera na innym porcie. |
| Provider jest pomijany | `foremankit status` pokazuje, którzy providerzy są aktywni, ich limity i wydatki; zablokowana sieć firmowa pomija providera automatycznie. |

## Budowa kodu

`router.py` jest cienką fasadą (`Router`); praca dzieje się w małych modułach:

| moduł | zadanie |
|---|---|
| `chain.py` | przejście po łańcuchu źródeł zadania (`ChainRun`), kolejność, odroczone źródła, błąd gdy nic nie odpowie |
| `candidates.py` | zamiana wpisu łańcucha (model w chmurze albo węzeł) w gotowego kandydata albo powód pominięcia |
| `caller.py` | jedno wywołanie: ponowienia, cooldowny, księgowanie zużycia, odrzucanie pustych i uciętych odpowiedzi |
| `probing.py`, `selection.py` | sondy węzłów i chmur, ranking węzłów, limity współbieżności |
| `postcheck.py` | komenda `verify` z poprawkami u tego samego modelu oraz `review` drugim modelem |
| `prompting.py`, `outputs.py`, `reasoning.py` | prompt i pliki kontekstu, zapis odpowiedzi, poziom rozumowania |
| `reports.py`, `formatting.py` | status, modele, statystyki, testy węzłów i ich tekstowe wyświetlanie |
| `records.py` | małe klasy danych przekazywane między krokami (`TaskRequest`, `CallContext`, `Candidate`, `Answer`) |

`Router(cfg, RouterEnv(...))`: środowisko (stan, log, sieć, zegar) podaje się jednym obiektem. `delegate(task, kind, sensitivity, **opcje)` przyjmuje opcje z `TaskRequest`.

## Testy

```bash
.venv/bin/python -m pytest -q        # Windows: .venv\Scripts\python.exe -m pytest -q
```
HTTP jest mockowany (`httpx.MockTransport`). Testy obejmują: routing, pomijanie providerów bez klucza, fallback po 429, ponowienie po 5xx, twardy budżet, koszt z cache hit, eskalację, limity rpd, niezależne on/off obu instancji Ollamy, priorytet CLI > env > config, private do remote przy wyłączonej local, odmowę dla publicznego adresu, `allow_private=false`, fail closed bez Ollamy, strażnika plików, profile, dry run i test end-to-end serwera MCP przez stdio. Testy modułów `nodes.py` i `verify.py` obejmują: ranking węzłów, priorytet CLI > env > config, adresy LAN, priorytet maszyny, limit danych, koszt, strażnik plików, profile, dry run i test end-to-end serwera MCP przez stdio.

---

## Status

Wczesny projekt jednego autora, rozwijany i testowany głównie na Windows; ścieżki dla Linuksa i macOS istnieją, ale są mniej sprawdzane. Interfejsy, klucze konfiguracji i wartości domyślne mogą się zmieniać między wersjami.

## Uwaga o danych i prywatności

Treść każdego delegowanego zadania trafia do providera wybranego przez router (np. DeepSeek, Google, Mistral, Groq), na warunkach tego providera. Darmowe plany mogą używać promptów do treningu. Używaj poziomów `sensitivity` opisanych wyżej, sekrety i dane osobowe trzymaj na `private` i sprawdź warunki każdego włączonego providera. Odpowiadasz za to, co wysyłasz.

## Znaki towarowe

foremankit jest niezależnym projektem i nie jest powiązany z Anthropic, DeepSeek, Google, Mistral, Groq, Cerebras, OpenRouter ani żadnym innym wymienionym providerem, ani przez nich wspierany. Claude i Claude Code są znakami towarowymi Anthropic.

## Licencja

[MIT](LICENSE). Copyright (c) 2026 Marcin Surunowicz.
