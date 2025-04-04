VERSION=$1
ncu --set full --kernel-name "softmax_kernel" --export "./report/softmax_v${VERSION}.ncu-rep" "./bin/softmax"
