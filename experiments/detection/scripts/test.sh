# Color
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'

IFS=$'\n'
function Fun_EvalCmd()
{
    cmd_list=$1
    i=0
    for cmd in ${cmd_list[*]}
    do
        ((i+=1))
        printf "${GREEN}\n${cmd}${NC}\n"
        eval $cmd
        exit_code=$?

        if [[ $exit_code = 0 ]]; then
            printf "${GREEN}[Success] ${cmd} ${NC}\n"
        else
            printf "${RED}[Failure: $exit_code] ${cmd} ${NC}\n"
            exit 1
        fi
    done
}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       Trap SIGINT and SIGTERM to stop child processes     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
killpids() {
    kpids=`jobs -p`
    for pid in ${kpids[*]}; do
        printf "${GREEN}kill -9 $pid${NC}\n"
        kill -9 $pid
    done
    exit 1
}
trap killpids SIGINT
trap killpids SIGTERM


# # # # # # # # # # # # # # # # # # # # #
#       python ./scripts/test.py        #
# # # # # # # # # # # # # # # # # # # # #
lCmdList=(
            # "CUDA_VISIBLE_DEVICES=0 python ./scripts/test.py ./config/hmnet_B3_yolox_regular_batch_relative.py --checkpoint checkpoint_59.pth.tar --fast --speed_test --test_chunks 2/2 &" \
            "CUDA_VISIBLE_DEVICES=$1 python ./scripts/test.py $2 --checkpoint $3 --fast --speed_test --test_chunks $4 &" \

         )
Fun_EvalCmd "${lCmdList[*]}"


# https://stackoverflow.com/questions/356100/how-to-wait-in-bash-for-several-subprocesses-to-finish-and-return-exit-code-0?page=1&tab=votes#tab-top
pids=`jobs -p`
for pid in ${pids[*]}; do
    printf "${GREEN}PID = $pid is running${NC}\n"
done

while true
do
    printf "${GREEN}while${NC}\n"
    for pid in $pids; do
        printf "${GREEN}1 wait $pid${NC}\n"
        wait $pid
        exit_code=$?
        printf "${GREEN}2 wait $pid${NC}\n"
        printf "${GREEN}$exit_code${NC}\n"
        if [ $exit_code != 0 ]
        then
            Fun_EvalCmd "${lCmdList[*]}"
        fi
    done
    pids=`jobs -p`
    count=0
    for pid in ${pids[*]}; do
        count=$count+1
        printf "${GREEN}PID = $pid is running${NC}\n"
    done
    if [ "$count" = "0" ]
    then
        exit 1
    fi
done