"cutechess-1.4.0-win64\cutechess-cli.exe" ^ -engine name="Guofish4" cmd="python" arg="playing/uci_wrapper.py" arg="--model" arg="models/guofish4/guofish4_25.6M_policy_final.pt" arg="--workers" arg="8" dir="." proto=uci tc=inf nodes=10000 timemargin=300000 ^ -engine name="Stockfish" cmd="stockfish-windows-x86-64-avx2.exe" dir="." proto=uci option.UCI_LimitStrength=true option.UCI_Elo=2800 option.Threads=1 tc=10+0.1 ^ -openings file="assets/8moves_v3.pgn" format=pgn order=sequential plies=16 ^ -resign movecount=3 score=600 ^ -draw movenumber=40 movecount=8 score=10 ^ -concurrency 7 ^ -rounds 25 -games 2 -repeat ^ -pgnout benchmarking/engine/games/v4/guofishv4_full_2_2.5_80.pgn


ordo-win64.exe -p guofish_eval_v4_CPUCT_1.00.pgn -A "Stockfish" -a 2500

ordo-win64.exe -p v4/guofish_eval_v4_CPUCT_1.00.pgn -A "Stockfish" -a 2500


"models/guofish4/guofish4_25.6M_policy_final.pt"

"models/guofish3/guofish3_25.6M_final_0.0691.pt"
