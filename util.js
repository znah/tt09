function prepareWASM(instance) {
    const type2class = {uint8_t: Uint8Array, int: Int32Array, uint32_t: Uint32Array, uint64_t: BigUint64Array};
    const prefix = '_len_';
    const exports = instance.exports;
    const outputs = {};
    const arrays = {}
    for (const key in exports) {
        if (!key.startsWith(prefix)) {
            if (!key.startsWith('_')) {
                outputs[key] = exports[key];
            }
            continue;
        }
        const [name, type] = key.slice(prefix.length).split('__');
        Object.defineProperty(outputs, name, {
            enumerable: true,
            get() {
                if (!(name in arrays) || arrays[name].buffer != exports.memory.buffer) { 
                    const ofs = exports['_get_'+name]();
                    const len = exports[key]();
                    if (!ofs) {
                        return null;
                    }
                    arrays[name] = new (type2class[type])(exports.memory.buffer, ofs, len);
                }
                return arrays[name];
            }
        });
    }
    return outputs;
}