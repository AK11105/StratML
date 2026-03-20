************OUTDATED***************
<!-- For windows,

Copy paste stratml.bat file into C:\Users\<name>\bin 

Open powershell in administrator mode,
    Run :-
        setx PATH "C:\Users\<name>\bin;$($env:Path)"
    Restart 

    Else - Manually add C:\Users\<name>\bin to user paths in env variables

Confirm by running -
        stratml init -->   
************OUTDATED***************


*LATEST*

1) Run this command in the same directory as the bat file. (/stratml/CLI)

    powershell -ExecutionPolicy Bypass -File install.ps1

2) Restart terminal or VSCode and Run this to confirm
   
    stratml init