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

## Kolejność
P1.1 → P1.3 → P3.1–P3.3 (równolegle, różne pliki) → P3.4–P3.5 → P4 → P3.6 → P5. P2 wtedy, gdy użytkownik ma dostęp do drugiego komputera (niezależnie od reszty).

## Dziennik
- 2026-10-13 (runda 2): P1.1 MCP: `_run_guarded` w server.py zwraca czytelny błąd, traceback w `<state>/mcp-errors.log` (+test). P1.3: `Installer.router_mode_active()` + ostrzeżenie po instalacji (+test); ustawienia VS Code tego komputera są już w trybie direct. P3.1–P3.3 przez Haiku (auto_checkpoint.run, install_hooks, panel.main, session_length_reminder.run); install_hooks sprawdzony skryptem porównawczym ze starą wersją (20 kombinacji, 0 różnic). 491 testów OK. Zostaje: panel.main ma 4 poziomy zagnieżdżenia (łańcuch elif), test_worker 22 linie, install.build_parser/run_uninstall 22 linie, inne drobne uwagi `foremankit check`. Nie zacommitowano.
- 2026-10-13: analiza (testy 489 OK, lista długich funkcji, otwarte punkty z pamięci), utworzenie planu. Nic nie zmieniono w kodzie.
