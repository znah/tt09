clear
FLAGS="-target wasm32-freestanding-musl -DWASM -lc -fno-entry --stack 65536 -O ReleaseSmall"
zig build-exe gates.c $FLAGS
ls -lh *.wasm