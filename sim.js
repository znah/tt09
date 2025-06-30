const type2class = {
    float: Float32Array,
    float32: Float32Array,
    int: Int32Array, 
    uint8:   Uint8Array,
    uint8_t: Uint8Array,
    uint16:   Uint16Array,
    uint16_t: Uint16Array,
    uint32:   Uint32Array,
    uint32_t: Uint32Array,
    uint64:   BigUint64Array,
    uint64_t: BigUint64Array,
};

function prepareWASM(instance) {
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

const range = n=>Array(n).fill(0).map((_,i)=>i);

function array2tex(glsl, array, format, tag, w=256) {
    const {chn, CpuArray} = TextureFormats[format];
    const n = array.length/chn;
    const h = Math.ceil(n/w);
    const data = new CpuArray(w*h*chn);
    data.set(array);
    return glsl({}, {size:[w,h], data, format, tag});
}

const VGA_SCREEN_W = 800;
const VGA_SCREEN_H = 525;
const VGA_ASPECT = VGA_SCREEN_W / VGA_SCREEN_H;

class VGASimulator {
    static async init(glsl) {
        const wasm = await WebAssembly.instantiateStreaming(fetch('gates.wasm'));
        return new VGASimulator(glsl, wasm);
    }

    constructor(glsl, wasm) {
        this.glsl = glsl;
        this.main = prepareWASM(wasm.instance);

        const MAX_W = VGA_SCREEN_W*5;
        this.screen_row = new Uint8Array(MAX_W*4);
        this.screen = glsl({}, {size:[MAX_W, VGA_SCREEN_H], tag:'screen'});
        this.min_steps_per_tick = 16;

        this.setupControls(glsl.gl.canvas);
    }

    load_circuit_bin(buf) {
        const bytes = new Uint8Array(buf);
        const jsonLength = bytes.indexOf(0);
        const data = JSON.parse(new TextDecoder().decode(bytes.slice(0, jsonLength)));
        for (const k in data) {
            const d = data[k];
            if (d.offset) {
                const len = d.shape.reduce((a, b) => a * b);
                d['a'] = new type2class[d.dtype](buf, d.offset, len)
            }
        }

        this.pin2wire = {};
        this.pin_pos = {}
        data.pins.forEach((name, i)=>{
            this.pin2wire[name] = data.pin_wires.a[i];
            const a = data.pin_pos.a;
            this.pin_pos[name] = [a[i*2], a[i*2+1]];
        });
        this.bbox = data.bbox;
        const [x0, y0, x1, y1] = this.bbox;
        this.view = {centerX:(x1+x0)/2, centerY:(y1+y0)/2, 
            log2zoom:0.0, pan:0.0, tilt:0.0, perspective:0.5,
            baseScale: 1.0/(x1-x0),
            wiresAlpha: 0.85,
            reveal:16.0,
            wireHeight:1.25,
            rotateMode: false,
        };
        this.geom  = {
            rects: array2tex(glsl, data.wire_rects.a, 'rgba16u', 'rects'),
            infos: array2tex(glsl, data.wire_infos.a, 'rg16u', 'infos'),
        };
        const out_pins = range(8).map(i=>`uo_out[${i}]`)
        this.out_wires = out_pins.map(pin=>this.pin2wire[pin]);
        this.main.gate_n[0] = data.luts.shape[0];
        for (const name of ['luts', 'inputs_start', 'inputs', 'outputs_start', 'outputs']) {
            this.main[name].set(data[name].a);
        }

        //this.out_pos = 
        // const rst_n = (tick>9) & 1; // ???
        // const [R1, G1, B1, vsync, R0, G0, B0, hsync] = this.out_wires;


        this.rayXY = [44, 33];
        this.tick = 0;
        this.substep = 0;
        this.steps_till_tick = this.min_steps_per_tick;
        this.startTime = Date.now();
        this.screen_width = VGA_SCREEN_W;

        this.screen_row.fill(0);
        this.main.state.fill(0);
        this.main.state[this.pin2wire['ui_in[7]']] = 1;
        let c=0;
        while (c=this.main.update_all()){ console.log(c)};
        this.updateState();
        this.updateState();  // 2x to make sure state story is valid

        // console.time('bench');
        // for (let i=0; i<100_000; ++i) while(!this.step());
        // console.timeEnd('bench');

    }

    sync_row() {
        const {glsl, rayXY} = this;
        const w = this.screen_row.length/4;
        const row = glsl({}, {data:this.screen_row, size:[w,1], tag:'row'});
        glsl({row, rayXY,
            VP:`XY.x,(rayXY.y+XY.y*0.5)/float(ViewSize.y)*2.0-1.0,0,1`,
            FP:`row(UV)`}, this.screen);
    }

    step(rowCallback=()=>{}) {
        const {main, tick, pin2wire, screen_row} = this;
        const clk = tick&1;
        const rst_n = (tick>9) & 1; // ???
        const [R1, G1, B1, vsync, R0, G0, B0, hsync] = this.out_wires;
        const S = main.state;
        const prev_hsync = S[hsync];
        const prev_vsync = S[vsync];
        main.set_signal(pin2wire.clk, clk);
        main.set_signal(pin2wire.rst_n, rst_n);
        const cycleDone = (main.run_wave()==0) && (this.steps_till_tick <= 0);
        this.steps_till_tick -= 1;
        if (cycleDone) {
            this.tick += 1;
            this.steps_till_tick = this.min_steps_per_tick;
        }

        let [x, y] = this.rayXY;
        const W = screen_row.length / 4;
        if (cycleDone && clk == 1) {
            if (x<W) {
                const p = x*4;
                screen_row[p+0] = S[R1]*170 + S[R0]*85;
                screen_row[p+1] = S[G1]*170 + S[G0]*85;
                screen_row[p+2] = S[B1]*170 + S[B0]*85;
            }
            x += 1;
        }
        if (prev_hsync && S[hsync]==0) {
            this.screen_width = Math.max(x, this.screen_width);
            x = 0;
            this.sync_row();
            this.screen_row.fill(0);
            y += 1;
            if (rowCallback) rowCallback(y-33);
        }
        if (prev_vsync && S[vsync]==0) {
            x = 0;
            y = 0;
        }
        this.rayXY = [x, y];
        return cycleDone;
    }

    multistep(step_n, rowCallback=()=>{}) {
        if (step_n >= 1) {
            let needBreak=false;
            for (let i=0; i<step_n && !needBreak; ++i) {
                sim.step(row=>{needBreak = rowCallback(row)});
            }
            this.updateState();
            this.substep = 0.0;
        } else {
            this.substep += step_n;
            if (this.substep >= 1.0) {
                sim.step(rowCallback);
                this.updateState();
                this.substep = 0.5*step_n;
            } 
        }
    }

    updateState() {
        const size = [256, Math.ceil(this.main.gate_n[0]/256)];
        this.state = glsl({}, {data:this.main.state, size, story:2, format:'r8u', tag:'state'});
        this.heat = glsl({}, {data:this.main.heat, size, format:'r32f', tag:'heat'});
        this.main.heat.fill(0);
    }

    draw_circuit(speed) {
        const {glsl, geom, state, heat, substep} = this;
    
        const smoothState = glsl({prev:state[1], next:state[0], heat, substep, speed,
            coef0: Math.pow(0.8, Math.min(1.0, speed)),
            coef1: Math.min(1.0/Math.pow(speed,0.5), speed), FP:`
            FOut = Src(I);
            float v0 = float(prev(I).r);
            float v1 = float(next(I).r);
            float h = heat(I).x;
            FOut.x = mix(v0, v1, substep);
            FOut.y = FOut.y*coef0 + h*coef1;
            FOut.z = max(FOut.z, min(1.0, h));
        `},{size:state.size, format:'rgba16f', story:2, tag:'smoothState'})

        const time = (Date.now()-this.startTime)/1000.0;            
        
        const viewAspect = glsl.gl.canvas.width / Math.max(glsl.gl.canvas.height,1);
        const [x0, y0, x1, y1]=this.bbox, [w, h] = [x1-x0, y1-y0];
        this.view.baseScale = w/h > viewAspect ? 1.0/w : 1.0/h/viewAspect;

        const view = {...this.view, Aspect:'x', Inc:`
        vec4 applyView(vec4 p) {
            vec2 center = vec2(centerX, centerY);
            p.xy = (p.xy-center)*rot2(pan);
            p.xyz *= 2.0*exp2(log2zoom)*baseScale;     // * (1.0/(1.0-sin(tilt)));
            p.xyz += float(ID.y%32)*5e-6; // less z-fighting
            p.yz *= rot2(tilt);
            p.zw = vec2(-p.z*0.01, 1.0-p.z*perspective);
            return p;
        }
        struct WireStyle {
            float value, heat, expand, glow, awake;
            vec4 color;
        };
        WireStyle wireStyle(int wire) {
            WireStyle s;
            ivec2 wireI = ivec2(wire&0xff, wire>>8);
            vec4 st = state(wireI);
            s.value = st.x;
            s.heat = st.y;
            s.awake = st.z;
            s.expand = 0.4 * s.heat / (1.0+s.heat);
            s.glow = 1.0 - min(1.0, 1.0/s.heat);
            s.color.rgb = mix(vec3(0.1, 0.4, 0.5), vec3(0.6, 0.4, 0.1), s.value);
            s.color.rgb *= 1.0+s.expand*4.0;
            s.color.rgb = mix( s.color.rgb, vec3(0.8, 0.1, 0.0), s.glow);
            s.color.a = wiresAlpha;
            return s;
        }
        float vignette(vec2 p) {
            p *= (1.0-p)*4.0;
            return p.x*p.y;
        }
        float vmax(vec3 p) { return max(p.x, max(p.y, p.z));}`};
        
        // draw clock
        const clkPos = this.pin_pos['clk'];
        const clkWire = this.pin2wire['clk'];
        glsl({DepthTest:1, ...view, time, clkPos, clkWire, state:smoothState[0],
            Grid:[6,1], Blend:'s*sa+d*(1-sa)', Face:'front', VP:`
            WireStyle ws = wireStyle(int(clkWire));
            varying vec4 color = ws.color;
            VPos.xyz = cubeVert(XY, ID.x)*(1.0+vec3(-ws.expand,ws.expand*3.,-ws.expand));
            VPos.xyz += vec3(clkPos, wireHeight*4.0) + vec3(0.0,2.0+float(ID.y),0.0); 
            VPos = applyView(VPos);
        `, FP:`
        float a = 0.8+0.2*vignette(UV);
        FOut = color*a;`});

        // draw gates
        const flipLayers = Math.cos(view.tilt) < -0.01;
        glsl({cellBox:this.bbox, ...geom, state:smoothState[0], 
            time, DepthTest:1, ...view, Face:'front', flipLayers,
            Grid:[6, ...geom.rects.size], Blend:'s*sa+d*(1-sa)', VP:`
            const int LAYER_IDX_0 = 16;
            const vec3 cmap[4] = vec3[4](
                vec3(0.6, 0.6, 0.6), // logic
                vec3(0.3, 0.3, 0.9), // clock gate
                vec3(0.9, 0.3, 0.3), // clock buf
                vec3(0.3, 0.9, 0.3)  // memory
            );

            ivec2 id = flipLayers ? (Grid.yz-ID.yz-1) : ID.yz;
            vec4 rect = vec4(rects(id)) / 65535.0;
            uvec4 info = infos(id);
            rect = mix(cellBox.xyxy, cellBox.zwzw, rect);
            int wire = int(info.x), layer = int(info.y);
            int wireLayerI = layer - LAYER_IDX_0 - 6;
            WireStyle ws = wireStyle(wire);
            float layerReveal = reveal;

            varying vec4 color = vec4(1.0);
            bool isVIA = (wireLayerI >= 0) && ((wireLayerI&1) == 0);
            vec3 p0 = vec3(rect.xy, -0.5);
            vec3 p1 = vec3(rect.zw, -0.001);
            if (layer >= LAYER_IDX_0) { // wires
                float viaH = 0.9;
                vec2 zrange = isVIA ? vec2(0.0, viaH) : vec2(viaH, 1.0);
                zrange = (float(wireLayerI>>1) + zrange)*wireHeight;
                p0.z = zrange.x; p1.z = zrange.y;
                vec3 pmean = (p0+p1)*0.5;
                vec3 d = p1-p0;
                d *= 0.5*vec3(equal(d, vec3(vmax(d))));
                //p0 = pmean-d; p1 = pmean+d;
                p0 -= ws.expand; p1 += ws.expand;
                layerReveal = layerReveal - float(wireLayerI+2);
                color = ws.color;
            } else { // cells
                color.rgb = cmap[layer] * (0.3+0.5*mix(ws.value, 0.5, ws.glow));
                layerReveal = clamp(layerReveal*10.0-hash(id.xyy).x*8.0, 0.0, 1.0);
                //layerReveal = min(layerReveal, ws.awake);
                if (layerReveal<1.0) {
                    float t = 1.0-layerReveal;
                    float lift = -t*t*(3.0-2.0*t) * 100.0;
                    p0.z += lift;
                    p1.z += lift;
                }
            }
            color.rgb *= float(ID.x==0)*0.5+0.9; // "lighting"

            // setup geometry
            vec3 cube = cubeVert(XY, ID.x);
            vec3 d = p1-p0, c=(p0+p1)*0.5;
            float m = vmax(d)*clamp(layerReveal, 0.0, 1.0);
            if (isVIA) {
                p1 = p0 + min(d, m);
            } else {
                d = min(d, m)*0.5;
                p0 = c-d; p1 = c+d;
            }   
            VPos.xyz = mix(p0, p1, cube*0.5+0.5);
            VPos = applyView(VPos);
            `, FP:`
            float a = 0.8 + 0.2*vignette(UV);
            FOut = color*a;`});
    }

    draw_screen(speed=1, fullsreen=false) {
        const {glsl, rayXY, screen} = this;

        // const inc= `
        // const vec2 justify = vec2(0.0, 1.0);
        // vec2 fit(vec2 p, vec4 box) {
        //     vec2 V=vec2(ViewSize), B=box.zw-box.xy;
        //     float ab = B.x/B.y, av = V.x/V.y;
        //     vec2 s = min(vec2(ab/av, av/ab), 1.0);
        //     vec2 q = 2.0/B*s;
        //     return q*(p-box.xy)-s + justify*(q-s+1.0);
        // }`;

        this.sync_row();
        const max_w = screen.size[0];
        const line_width = this.screen_width;
        const WH = [this.screen_width, VGA_SCREEN_H];
        glsl({vgaAspect:VGA_ASPECT, WH, max_w, rayXY, screen, fullsreen, rayTail:Math.sqrt(speed), 
            Blend:'d*(1-sa)+s', VP:`
            float viewAspect = float(ViewSize.x) / float(ViewSize.y);
            vec2 p = vec2(XY.x, -XY.y);
            float aspectRatio = vgaAspect/viewAspect;
            if (aspectRatio < 1.0) {
                p.x *= aspectRatio;
            } else {
                p.y /= aspectRatio;
            }
            if (!fullsreen) {
                p = p*0.5 + vec2(-0.5,0.5);
            }
            VPos.xy = p;
            `, FP:`
            vec4 c = screen(vec2(UV.x*WH.x/max_w, UV.y));
            vec2 a = cos(XY*PI/2.);
            float alpha = 1.0-pow(1.0-a.x*a.y, 8.0);
            FOut = vec4(c.rgb, alpha*0.5);
            vec2 pixelPos = UV*WH;
            vec2 dray = (pixelPos - rayXY)/4.0;
            if (dray.x < 0.0) {
                dray.x /= rayTail;
            }
            if (dray.y > 0.0) {
                FOut.rgb *= smoothstep(0.0, 10.0, dray.y);
            }
            FOut.rgb += exp(-dot(dray,dray))*vec3(1.,1.,0.3);
            `});
    }

    _handleMove(dx, dy, isRotate) {
        const {view} = this;
        if (isRotate) {
            view.pan += dx;
            view.tilt += dy;
        } else {
            const speed = Math.pow(2.0, -view.log2zoom)/view.baseScale;
            const s = Math.sin(view.pan), c = Math.cos(view.pan);
            view.centerX += (c*dx + s*dy)*speed;
            view.centerY += (s*dx - c*dy)*speed;
        }
    }

    setupControls(canvas) {
        const scrollHandler = (e)=>{
            e.preventDefault();
            const {view} = this;
            const dx=e.deltaX*0.001, dy=e.deltaY*0.001;
            if (e.ctrlKey) {
                view.log2zoom -= dy*10.0;
                view.log2zoom = Math.max(0.0, view.log2zoom);
            } else {
                this._handleMove(dx, dy, e.shiftKey);
            }
        };
        canvas.addEventListener('wheel', scrollHandler, {passive: false});

        const dragHandler = (e) => {
            if (e.buttons !== 1) return;
            e.preventDefault();
            let dx = -e.movementX * 0.001, dy = -e.movementY * 0.001;
            const isRotate = e.shiftKey || this.view.rotateMode;
            if (isRotate) { dx *= 2; dy *= 2; }
            this._handleMove(dx, dy, isRotate);
        };
        canvas.addEventListener('pointerdown', e => {
            if (!e.isPrimary || e.button !== 0) return;
            canvas.setPointerCapture(e.pointerId);
            canvas.addEventListener('pointermove', dragHandler);
            canvas.style.cursor = 'grabbing';
        });
        ['pointerup', 'pointercancel'].forEach(name=>canvas.addEventListener(name, e => {
            if (!e.isPrimary) return;
            canvas.removeEventListener('pointermove', dragHandler);
            canvas.style.cursor = 'grab';
        }));        
    }

}
