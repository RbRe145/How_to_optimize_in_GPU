VERSION=$1
ncu --set full --kernel-name "reduce${VERSION}" -f --export "./report/reduce${VERSION}.ncu-rep" "./bin/reduce_v${VERSION}"