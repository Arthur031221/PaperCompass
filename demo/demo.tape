# VHS tape: https://github.com/charmbracelet/vhs
# Render with: vhs demo/demo.tape (run from the repo root)
Output demo/demo.gif

Set Shell "zsh"
Set FontSize 18
Set Width 1200
Set Height 720
Set Theme "Dracula"
Set TypingSpeed 40ms

Type "papercompass --help"
Enter
Sleep 2s

Type "papercompass import demo/sample_seeds.txt --data-dir /tmp/papercompass-demo --json"
Enter
Sleep 4s

Type "papercompass ingest --categories cs.LG --max-per-category 40 --data-dir /tmp/papercompass-demo --json"
Enter
Sleep 6s

Type "papercompass serve --port 8931 --data-dir /tmp/papercompass-demo"
Enter
Sleep 3s
