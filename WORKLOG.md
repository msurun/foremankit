# ForemanKit – dziennik prac i plan poprawek

Plik do śledzenia między sesjami. Status: `[ ]` do zrobienia, `[~]` w toku, `[x]` zrobione, `[?]` czeka na dane/decyzję.
Po każdej sesji: dopisz wpis w „Dziennik" i zaktualizuj statusy. Decyzje i niezweryfikowane rzeczy: patrz też pamięć `project-foremankit-state-2026-10-13`.

## Stan wyjściowy (2026-10-13)
- main @ 65757e4, drzewo czyste, 489 testów przechodzi (~31 s).
- Funkcje powyżej standardu (>20 linii / >4 parametrów): `hooks/auto_checkpoint.py:run` (80), `install.py:install_hooks` (66), `foremankit/panel.py:main` (48), `hooks/session_length_reminder.py:run` (44), `foremankit/server.py:delegate` (11 parametrów – to interfejs MCP, raczej zostawić), `pairing.py:__init__` (7 parametrów).
- Największe pliki: install.py (1114 linii), panel.py (731), cli.py (332).

## Plan prac

### P1 – błędy i ryzyka (najpierw)
- [x] **P1.1** MCP `delegate` zwrócił raz gołe „Error executing tool". Dodać w `server.py` łapanie wyjątków z logiem (typ + traceback do pliku logu) i czytelnym komunikatem; test z wymuszonym wyjątkiem. Reprodukcja: brak – log jest właśnie po to.
- [~] **P1.2** (wdrożone, niezweryfikowane na żywo: `side_requests_to_cloud` w gateway.py, żądania /v1/messages bez narzędzi idą do chmury jako `side_request_model`; hipoteza użytkownika: „Error executing tool" = tryb auto Claude Code dostaje odpowiedź Qwena zamiast Sonneta; log bramki z 13.10 nie zawiera błędu, więc brak potwierdzenia — sprawdzić po restarcie bramki w trybie router + auto; `handle` ma teraz 27 linii, do podziału) Klasyfikator auto-mode (żądanie bez narzędzi) trafia do lokalnego Qwena (1 slot) i wygasa po 150 s. Po zebraniu logu bramki: kierować żądania poboczne bez narzędzi do chmury (`gateway.py`). `[?]` czeka na jedną reprodukcję.
- [x] **P1.3** Instalator zostawił VS Code w trybie router; użytkownik ma kliknąć „Stop (direct mode)". Sprawdzić, czy `install.py` może to wykryć i wypisać jasną uwagę po instalacji.

### P2 – weryfikacja rzeczy nigdy nie sprawdzonych
- [ ] **P2.1** Parowanie dwóch prawdziwych komputerów + uruchomienie `llama/pair-firewall.ps1` z UAC.
- [ ] **P2.2** Przycisk `Test` na prawdziwym zdalnym workerze; zdalny orchestrator po LAN end-to-end.
- [ ] **P2.3** Wygląd panelu w VS Code (dotąd tylko zrzuty użytkownika i test fałszywego DOM) – przejrzeć po restarcie VS Code (`FOREMANKIT_LANG=pl`).

### P3 – zgodność z własnym standardem kodu (małe, mechaniczne → agenci Haiku, po jednym pliku)
- [x] **P3.1** Rozbić `auto_checkpoint.run` (80 linii) na funkcje ≤20 linii; testy już są? sprawdzić `tests/` przed zmianą.
- [x] **P3.2** Rozbić `install_hooks` (66) i `panel.main` (48).
- [x] **P3.3** Rozbić `session_length_reminder.run` (44).
- [ ] **P3.4** `pairing.__init__`: 7 parametrów → mały dataclass konfiguracji.
- [ ] **P3.5** Zainstalować ruff + vulture jako opcjonalne zależności dev; uruchomić, naprawić tylko oczywiste.
- [ ] **P3.6** Podzielić `install.py` (1114 linii) na moduły (hooki, ustawienia VS Code, rozszerzenie, odinstalowanie); zrobić dopiero po P3.2, osobno, z `tests/test_install.py` i `test_uninstall_full.py` jako siatką bezpieczeństwa.

### P4 – ulepszenia użytkowe (z listy pomysłów)
- [ ] **P4.1** Ręczne pole adresu w parowaniu (gdy discovery UDP nie działa).
- [ ] **P4.2** Przycisk w panelu otwierający `%TEMP%\llama-server.log.err`.
- [ ] **P4.3** Wykrywanie `claude.exe` poza PATH (ścieżka w `~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/`) i podpowiedź w instalatorze/README.
- [ ] **P4.4** Licznik delegacji: raport „ile linii napisano samemu vs zdelegowano" w `foremankit stats` (dane z hooka nudge już są).

### P5 – dokumentacja i testy
- [ ] **P5.1** README (EN/PL): sekcja o trybie direct vs router, parowaniu i zasadach firewalla – zsynchronizować po zmianach P1–P4.
- [ ] **P5.2** Test regresyjny dla każdej naprawy z P1 (reprodukcja najpierw).

## Propozycja planu rozwoju (opus-advisor, 2026-10-14) – do decyzji użytkownika
Wniosek: ForemanKit rozbudowuje infrastrukturę, ale nie mierzy celu (oszczędność limitu Claude). Wąskie gardło to zachowanie Claude (0 delegacji w sesji 3600 linii), nie routing.
Brakuje pomiaru: `calls.jsonl` zapisuje tylko wywołania providerów (caller.py:65); wyniki verify/review nie trafiają do logu (postcheck.py:66, 164); `saved_usd` (reports.py:209-215) liczy każdy zdelegowany token jako zaoszczędzony token Sonneta, bez kosztu nadzoru i odrzuceń; bramka loguje tylko na stdout (gateway.py:86-90).
Kierunki wg wartości/ryzyka:
1. **Pomiar** (ryzyko niskie, warunek reszty): zdarzenie `outcome` przy każdym `delegate` (verify passed/rounds, blokery review, czy plik przetrwał); parser `usage` z transkryptu Claude (tokeny na sesję; cache raportować osobno); w `foremankit stats`: udział zdelegowanych tokenów, koszt na zaakceptowany wynik, odsetek odrzuceń na model.
2. **Odchudzić globalny CLAUDE.md** (~70 linii czytanych w każdej turze) do checklisty; reszta do pliku czytanego na żądanie. Jedyna pewna oszczędność od ręki.
3. **Tańszy nadzór**: verify + conventions domyślnie dla `bulk_code`, podgląd 600 znaków (routing.yaml:118) ma wystarczać, żeby Claude nie czytał `.agent-out`; zmierzyć, ile razy czyta.
4. **Lokalny orchestrator w trybie router**: największy potencjał (0 tokenów Claude), najwyższe ryzyko. Najpierw zweryfikować P1.2 na żywo, potem wskaźnik sukcesu z benchmarks/agent_task.py.
5. **Zamrozić wiele maszyn** (pairing, zdalny orchestrator, firewall = P2, nigdy niesprawdzone) do czasu regularnego użycia drugiej maszyny.
Wyciąć/odłożyć: P3.6 (podział install.py), P3.4 i P4.1-P4.3 do czasu P2, nowe funkcje panelu ponad on/off i status, `mistral`/`groq` w łańcuchu bulk_code jeśli statystyki pokażą znikomy udział, obecne `saved_usd` -> koszt na zaakceptowany wynik.
Kolejność sesji: S1 test P1.2 na żywo (10 min) + logowanie `outcome` + nowe kolumny stats + parser transkryptu (zdelegować, bulk_code z verify); S2 przyciąć CLAUDE.md, tydzień normalnej pracy, analiza odrzuceń i kosztów na model, przycięcie łańcuchów; S3 decyzja na podstawie benchmarku: rozwijać orchestrator albo zamrozić.
Ryzyka: format transkryptu Claude może się zmienić (parser odporny na brak pól + test na prawdziwej próbce); „zaakceptowany" to przybliżenie (zweryfikować ręcznie na 10 przypadkach z git diff).

## Kolejność
P1.1 → P1.3 → P3.1–P3.3 (równolegle, różne pliki) → P3.4–P3.5 → P4 → P3.6 → P5. P2 wtedy, gdy użytkownik ma dostęp do drugiego komputera (niezależnie od reszty).

## Dziennik
- 2026-10-14: review wtyczki przez DeepSeek Pro (2 wywołania, $0,06; escalate niepotrzebne, hard_code i tak idzie do deepseek:pro) i przez lokalnego Qwena w 5 ratach (kontekst serwera 65536). Zweryfikowane i naprawione: apostrof w argumentach PowerShell (`ps_quote` w panel.py), `pairTimer` bez sprzątania, nonce CSP z `crypto`. Fałszywe: `run_quick` KeyError (argparse choices), timeout `readSlots`, CSP „za restrykcyjne". Qwen na ratach: mało sygnału (bloki bez kontekstu innych części dają fałszywe „undefined method"); drobne uwagi o nieatomowych zapisach (`set_orchestrator`, `set_direct_mode`) nie zweryfikowane. Test session_hook uodporniony na FOREMANKIT_LANG=pl. Plan opusa zapisany powyżej. Qwen3.6-35B działa lokalnie jako robotnik (port 8080).
- 2026-10-13 (runda 2): P1.1 MCP: `_run_guarded` w server.py zwraca czytelny błąd, traceback w `<state>/mcp-errors.log` (+test). P1.3: `Installer.router_mode_active()` + ostrzeżenie po instalacji (+test); ustawienia VS Code tego komputera są już w trybie direct. P3.1–P3.3 przez Haiku (auto_checkpoint.run, install_hooks, panel.main, session_length_reminder.run); install_hooks sprawdzony skryptem porównawczym ze starą wersją (20 kombinacji, 0 różnic). 491 testów OK. Zostaje: panel.main ma 4 poziomy zagnieżdżenia (łańcuch elif), test_worker 22 linie, install.build_parser/run_uninstall 22 linie, inne drobne uwagi `foremankit check`. Nie zacommitowano.
- 2026-10-13: analiza (testy 489 OK, lista długich funkcji, otwarte punkty z pamięci), utworzenie planu. Nic nie zmieniono w kodzie.
