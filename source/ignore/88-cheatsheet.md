# cheatsheet for markdown syntax

## Markdown Shortcuts

모든 명령어
Ctrl-m 두 번

## Paste Image

Ctrl+Alt+V

한 줄 씩 띄어서

![test](paste.png){#fig:label}

![test2](paste2.png){width=20%}

![test3](paste2.png){height=20%}

<div id="fig:">

![subfigure 1 caption](paste.png){height=20%} \qquad
![subfigure 2 caption](paste2.png){height=20% #fig:testB}

Caption of figure
</div>

[@fig:testB]처럼 인용


a   b   c
--- --- ---
1   2   3
4   5   6

: Caption {#tbl:label}

## Extension: smart

Interpret straight quotes as curly quotes, --- as em-dashes, -- as en-dashes, and ... as ellipses.

## section header {#sec:cheat}

[@sec:cheat](장)

[-@sec:cheat]절

\newpage
