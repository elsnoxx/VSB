import subprocess
import filecmp
import difflib


source_file = input("Zadejte název zdrojového souboru (např. program.c): ")
# kompilace
compile_result = subprocess.run(["gcc", source_file, "-o", "main", "-g", "-fsanitize=address"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
compile_output = compile_result.stderr



Pocet_OK = 0

# kontrola kompilace
if compile_result.returncode == 0:
    print("\x1b[32mKompilace proběhla úspěšně\x1b[0m")
    Pocet_OK+=1
else:
    print("\x1b[31mKompilace selhala. Chybový výstup:\x1b[0m")
    print(compile_output)




# Spuštění programu a získání výstupu
program_output = subprocess.run(["./main"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
with open("output.txt", "w") as f:
    f.write(program_output.stdout)

# Očekávaný výstup

with open("tests/test-big/stdout", "r") as expected_output:
    # Porovnání výstupu s očekávaným výstupem
    if filecmp.cmp("output.txt", "/tests/test-big/stdout"):
        print("\x1b[32mTest proběhl úspěšně\x1b[0m")
    else:
        print("\x1b[31mTest selhal\x1b[0m")
        print(program_output.stdout)



print("{}/14".format(Pocet_OK))