# System Oceny Jakości Kodu Oparty na Wnioskowaniu Rozmytym (Interval Type-2 TSK)

Dokumentacja techniczna projektu wykorzystującego logikę rozmytą typu 2 do automatycznej oceny jakości kodu źródłowego Python.

---

## Spis Treści
1. [Wstęp i Cel Projektu](#1-wstęp-i-cel-projektu)
2. [Metryki Wejściowe](#2-metryki-wejściowe)
3. [Projekt Systemu Rozmytego (Fuzzy Logic Design)](#3-projekt-systemu-rozmytego-fuzzy-logic-design)
    - [Zmienne Lingwistyczne](#zmienne-lingwistyczne)
    - [Funkcje Przynależności (Membership Functions)](#funkcje-przynależności-membership-functions)
4. [Baza Reguł i Wnioskowanie](#4-baza-reguł-i-wnioskowanie-tsk)
5. [Implementacja i Technologie](#5-implementacja-i-technologie)
6. [Instrukcja Uruchomienia](#6-instrukcja-uruchomienia)
7. [Przykłady Działania i Wyniki](#7-przykłady-działania-i-wyniki)
8. [Wnioski i Możliwy Rozwój](#8-wnioski-i-możliwy-rozwój)

---

## 1. Wstęp i Cel Projektu

## 2. Metryki Wejściowe

System podejmuje decyzje na podstawie dwóch kluczowych wskaźników wyekstrahowanych z kodu:

### 2.1. Złożoność Cyklomatyczna (Cyclomatic Complexity - CC)
### 2.2. Gęstość Węzłów AST (AST Node Density)
## 3. Projekt Systemu Rozmytego (Fuzzy Logic Design)

System oparty jest na **Interval Type-2 Fuzzy Logic System (IT2FLS)** w modelu **Takagi-Sugeno-Kang (TSK)**. Wybór tego rozwiązania podyktowany był koniecznością modelowania niepewności, która jest nierozerwalnie związana z subiektywną oceną jakości kodu.

W przeciwieństwie do klasycznych systemów rozmytych (Type-1), gdzie stopień przynależności jest precyzyjną wartością punktową (np. 0.5), IT2FLS wykorzystuje tzw. **Ślad Niepewności (Footprint of Uncertainty - FOU)**, ograniczony przez górną (Upper MF) i dolną (Lower MF) funkcję przynależności. Dzięki temu system operuje na przedziałach wartości, co znacznie lepiej oddaje naturę ludzkiego rozumowania, w którym definicje pojęć takich jak „złożony” czy „czytelny” są często nieostre i zależne od kontekstu.

Główne zalety tego podejścia w kontekście projektu to zdolność do efektywnego radzenia sobie z niejednoznacznościami oraz zapewnienie płynniejszej powierzchni sterowania. Eliminuje to nagłe skoki oceny przy niewielkich zmianach metryk wejściowych, co potwierdzono w testach – system skutecznie rozróżniał niuanse kodu (np. przypadki o wysokiej gęstości, ale niskiej złożoności), unikając skrajnych ocen dla przypadków granicznych.

Implementacja została zrealizowana w języku Python przy użyciu biblioteki `pyit2fls`. Zdefiniowano zbiory rozmyte typu 2 (IT2FS) za pomocą funkcji trapezoidalnych, a sterownik TSK skonfigurowano z wykorzystaniem iloczynowej t-normy oraz s-normy Einsteina. Ostateczny, ostry wynik (Crisp Output) uzyskiwany jest poprzez algorytm redukcji typu.

### Zmienne Lingwistyczne

#### Wejście 1: Density (Gęstość)
- **Uniwersum:** `0 - 40` węzłów/linię.
- **Zbiory:**
    1. `Low` (Kod rzadki/rozwlekły)
    2. `Optimal` (Kod zrównoważony/czytelny)
    3. `High` (Kod zbity/trudny kognitywnie)

#### Wejście 2: Complexity (Złożoność CC)
- **Uniwersum:** `0 - 50`.
- **Zbiory:**
    1. `Low` (Kod prosty)
    2. `Medium` (Kod przeciętny)
    3. `High` (Kod trudny/zły)

### Funkcje Przynależności (Membership Functions)
Do modelowania niepewności wykorzystano funkcje trapezoidalne z określonym *Upper Membership Function (UMF)* i *Lower Membership Function (LMF)*.

## 4. Baza Reguł i Wnioskowanie (TSK)

Macierz reguł ($3 \times 3$) definiuje funkcję wyjścia (Quality Score) w zależności od kombinacji wejść.

| Complexity \ Density | Low (Rzadki) | Optimal (Optymalny) | High (Zbity) |
|----------------------|--------------|--------------------|--------------|
| **Low (Prosty)** | *Reguła 1*   | *Reguła 2* | *Reguła 3*<br> |
| **Medium (Średni)** | *Reguła 4*   | *Reguła 5* | *Reguła 6* |
| **High (Trudny)** | *Reguła 7*   | *Reguła 8* | *Reguła 9*<br> |

### Funkcje Wniosku (Consequents)
W modelu TSK wyjście nie jest zbiorem rozmytym, lecz funkcją liniową postaci:
$$f(x) = C + w_1 \cdot Density + w_2 \cdot Complexity$$

## 5. Implementacja i Technologie

Projekt zrealizowano w języku **Python 3.x**.

**Kluczowe biblioteki:**
* `pyit2fls`: Silnik logiki rozmytej typu 2.
* `radon`: Ekstrakcja metryk surowych i CC.
* `ast` (Standard Library): Analiza drzewa składniowego do obliczenia gęstości.
* `numpy` & `matplotlib`: Obliczenia numeryczne i wizualizacja powierzchni sterowania 3D.

## 6. Instrukcja Uruchomienia

### Wymagania
* Python 3.8+
* Zainstalowane pakiety z `requirements.txt`

