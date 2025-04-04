VERSION=$1
ncu --set full --kernel-name "softmax_cuda" --export "./report/transpose_v${VERSION}.ncu-rep" "./bin/transpose_v${VERSION}"