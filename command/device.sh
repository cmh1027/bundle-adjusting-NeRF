DEVICE=$1

if [ ${DEVICE} -eq 4 ];
then
    sh command/command.sh ${DEVICE} ship
    sh command/command.sh ${DEVICE} material
fi
if [ ${DEVICE} -eq 5 ];
then
    sh command/command.sh ${DEVICE} ficus
    sh command/command.sh ${DEVICE} hotdog
fi
if [ ${DEVICE} -eq 6 ];
then
    sh command/command.sh ${DEVICE} lego
    sh command/command.sh ${DEVICE} drums
fi
if [ ${DEVICE} -eq 7 ];
then
    sh command/command.sh ${DEVICE} mic
    sh command/command.sh ${DEVICE} chair
fi