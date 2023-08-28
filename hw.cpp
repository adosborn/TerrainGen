#define COW_CRASH_ON_FLOATING_POINT_EXCEPTIONS
#define _CRT_SECURE_NO_WARNINGS
#include "snail.cpp"
#include "cow.cpp"
#include "_cow_supplement.cpp"
#include <time.h>

//noise 1st implimentation - simplex
//
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
// 

vec3 mod289(vec3 x) {
    return x - V3(floor(x.x * (1.0 / 289.0)), floor(x.y * (1.0 / 289.0)), floor(x.z * (1.0 / 289.0))) * 289.0;
}

vec2 mod289(vec2 x) {
    return x - V2(floor(x.x * (1.0 / 289.0)), floor(x.y * (1.0 / 289.0))) * 289.0;
}

vec3 permute(vec3 x) {
    vec3 new_vec = (x * 34.0) + V3(10.0,10.0,10.0);
    return mod289(V3(new_vec.x*x.x, new_vec.y*x.y, new_vec.z*x.z));
}

float snoise(vec2 v) {
    vec4 C = V4((3.0-sqrt(3.0))/6.0, 0.5*sqrt(3.0)-1.0, -1.0 + 2.0 * ((3.0-sqrt(3.0))/6.0), 1.0 / 41.0);

    //first corner
    vec2 i = V2(floor(v.x + dot(v, V2(C.y, C.y))), floor(v.y + dot(v, V2(C.y, C.y))));
    vec2 x0 = V2(v.x - i.x + dot(i, V2(C.x, C.x)), v.y - i.y + dot(i, V2(C.x, C.x)));

    //other courners
    vec2 i1;
    if(x0.x > x0.y) i1 = V2(1.0, 0.0);
    else i1 = V2(0.0, 1.0);
    vec4 x12 = V4(x0.x + C.x - i1.x, x0.y + C.x - i1.y, x0.x + C.z, x0.y + C.z);

    //permutations
    i = mod289(i);
    vec3 temp = permute(V3(0, i.y + i1.y, 1));
    vec3 p = permute(V3(temp.x + i.x, temp.y + i1.x, temp.z + 1));
    vec3 temp2 = V3(dot(x0,x0), dot(V2(x12.x, x12.y), V2(x12.x, x12.y)), dot(V2(x12.z, x12.w), V2(x12.z, x12.w)));
    vec3 m = V3(fmax(0.5 - temp2.x, 0), fmax(0.5 - temp2.y, 0), fmax(0.5 - temp2.z, 0));
    m = V3(m.x*m.x, m.y*m.y, m.z*m.z);
    m = V3(m.x*m.x, m.y*m.y, m.z*m.z);

    //gradients
    vec3 x = 2.0 * V3(fract(p.x * C.w), fract(p.y * C.w), fract(p.z * C.w)) - V3(1.0, 1.0, 1.0);
    vec3 h = V3(abs(x.x), abs(x.y), abs(x.z)) - V3(0.5, 0.5, 0.5);
    vec3 ox = V3(floor(x.x + 0.5), floor(x.y + 0.5), floor(x.z + 0.5));
    vec3 a0 = x - ox;

    //normalize
    vec3 temp3 = V3(a0.x * a0.x, a0.y * a0.y, a0.z * a0.z) + V3(h.x * h.x, h.y * h.y, h.z * h.z);
    m = V3(m.x * (1.79284291400159 - (0.85373472095314 * temp3.x)),
        m.y * (1.79284291400159 - (0.85373472095314 * temp3.y)), 
        m.z * (1.79284291400159 - (0.85373472095314 * temp3.z)));

    //compute final value at p 
    vec3 g;
    g.x = (a0.x * x0.x) + (h.x * x0.y);
    g.y = (a0.y * x12.x) + (h.y * x12.y);
    g.z = (a0.z * x12.z) + (h.z * x12.w);
    return 130.0 * dot(m,g);
}


//noise 2nd implimentation - perlin
// GLSL textureless classic 2D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-08-22
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
// https://github.com/stegu/webgl-noise
//
vec4 v = V4(1.0,1.0,1.0,1.0);
vec4 mod289(vec4 x)
{
  return x - V4(floor(x.x * (1.0 / 289.0)), floor(x.y * (1.0 / 289.0)), floor(x.z * (1.0 / 289.0)), floor(x.w * (1.0 / 289.0))) * 289.0;
}

vec4 permute(vec4 x)
{
  return mod289(((x*34.0)+V4(10.0 * x.x, 10.0 * x.y, 10.0 * x.z, 10.0 * x.w)));
}

vec4 taylorInvSqrt(vec4 r)
{
  return V4(1.79284291400159, 1.79284291400159, 1.79284291400159, 1.79284291400159) - 0.85373472095314 * r;
}

vec2 fade(vec2 t) {
  return V2(pow(t.x,3)*(t.x*(t.x*6.0-15.0)+10.0), pow(t.y,3)*(t.y*(t.y*6.0-15.0)+10.0));
}

// Classic Perlin noise
vec4 xyxy (vec2 p){
    return V4(p.x, p.y, p.x, p.y);
}
vec4 xzxz (vec4 p){
    return V4(p.x, p.z, p.x, p.z);
}
vec4 yyww (vec4 p){
    return V4(p.y, p.y, p.w, p.w);
}
vec4 v4abs(vec4 p){
    return V4(abs(p.x), abs(p.y), abs(p.z), abs(p.w));
}
vec4 v4floor (vec4 p){
    return V4(floor(p.x), floor(p.y), floor(p.z), floor(p.w));
}
vec4 v4fract(vec4 p){
    return V4(fract(p.x), fract(p.y), fract(p.z), fract(p.w));
}
vec2 v2mix(vec2 a, vec2 b, float wb){
    return a * (1.0 - wb) + (b * wb);
} 
float mix(float a, float b, float wb){
    return a * (1.0 - wb) + (b * wb);
} 
vec4 v4divide(vec4 a, vec4 b){
    return V4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w);
}
vec4 v4mult(vec4 a, vec4 b){
    return V4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
vec4 v4mod(vec4 a, vec4 b){
    return a - v4mult(b, v4floor(v4divide(a,b)));
}
float cnoise(vec2 P)
{
vec4 Pi = v4floor(xyxy(P)) + V4(0.0, 0.0, 1.0, 1.0);
vec4 Pf = v4fract(xyxy(P)) - V4(0.0, 0.0, 1.0, 1.0);
Pi = mod289(Pi); // To avoid truncation effects in permutation
vec4 ix = xzxz(Pi);
vec4 iy = yyww(Pi);
vec4 fx = xzxz(Pf);
vec4 fy = yyww(Pf);

vec4 i = permute(permute(ix) + iy);
vec4 gx = v4fract(i * (1.0 / 41.0)) * 2.0 - v;
vec4 gy = v4abs(gx) - v/2;
vec4 tx = v4floor(gx + v/2);
gx = gx - tx;

vec2 g00 = V2(gx.x,gy.x);
vec2 g10 = V2(gx.y,gy.y);
vec2 g01 = V2(gx.z,gy.z);
vec2 g11 = V2(gx.w,gy.w);

vec4 norm = taylorInvSqrt(V4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
g00 *= norm.x;  
g01 *= norm.y;  
g10 *= norm.z;  
g11 *= norm.w;  

float n00 = dot(g00, V2(fx.x, fy.x));
float n10 = dot(g10, V2(fx.y, fy.y));
float n01 = dot(g01, V2(fx.z, fy.z));
float n11 = dot(g11, V2(fx.w, fy.w));

vec2 fade_xy = fade(V2(Pf.x, Pf.y));
vec2 n_x = v2mix(V2(n00, n01), V2(n10, n11), fade_xy.x);
float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
return 2.3 * n_xy;
}

// Classic Perlin noise, periodic variant
float pnoise(vec2 P, vec2 rep)
{
    vec4 Pi = v4floor(xyxy(P)) + V4(0.0, 0.0, 1.0, 1.0);
    vec4 Pf = v4fract(xyxy(P)) - V4(0.0, 0.0, 1.0, 1.0);
    Pi = v4mod(Pi, xyxy(rep));
    Pi = mod289(Pi); // To avoid truncation effects in permutation
    vec4 ix = xzxz(Pi);
    vec4 iy = yyww(Pi);
    vec4 fx = xzxz(Pf);
    vec4 fy = yyww(Pf);

    vec4 i = permute(permute(ix) + iy);
    vec4 gx = v4fract(i * (1.0 / 41.0)) * 2.0 - v;
    vec4 gy = v4abs(gx) - v/2;
    vec4 tx = v4floor(gx + v/2);
    gx = gx - tx;

    vec2 g00 = V2(gx.x,gy.x);
    vec2 g10 = V2(gx.y,gy.y);
    vec2 g01 = V2(gx.z,gy.z);
    vec2 g11 = V2(gx.w,gy.w);

    vec4 norm = taylorInvSqrt(V4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
    g00 *= norm.x;  
    g01 *= norm.y;  
    g10 *= norm.z;  
    g11 *= norm.w;  

    float n00 = dot(g00, V2(fx.x, fy.x));
    float n10 = dot(g10, V2(fx.y, fy.y));
    float n01 = dot(g01, V2(fx.z, fy.z));
    float n11 = dot(g11, V2(fx.w, fy.w));

    vec2 fade_xy = fade(V2(Pf.x, Pf.y));
    vec2 n_x = v2mix(V2(n00, n01), V2(n10, n11), fade_xy.x);
    float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}

struct grid {
    int count = 0; 
    int tri_count = 0;
    int length;
    int width;
    double offset_w;
    double offset_l;
    double height_scale;
    vec3* pos;
    int3* tri_indices;
    vec3* colors;
    int octaves; 
    double persistance;
    double lacunarity;
    int mode; 
};

vec3 color_gen(double h, vec3* colors, int cols_in_palette, double max, double min) {
    vec3 col = V3(1,1,1);
    //convert h to t for lerp
    double h_min = min; double h_max = max;
    double percent = (h + (h_max - h_min)/2) / (h_max - h_min);

    double cur_div = 0;
    for(int i = 0; i < (cols_in_palette - 1); i++){
        cur_div += 1/(double)(cols_in_palette-1);
        if(percent < cur_div){
            col = LERP(percent, colors[i], colors[i+1]);
            break;
        }
    }
    return col;
}

double octave_noise(vec2 cords, int octaves, double persistance, double lacunatrity, int mode, double width, double length){
    double noise_height = 0;
    double freq = 1;
    double amplitude = 1;
    for(int i = 0; i < octaves; i++){
        double value = 0;
        if(mode == 1) value = snoise(freq * cords);
        if(mode == 2) value = cnoise(freq * cords);
        if(mode == 3) value = pnoise(freq * cords, freq * V2(cords.x + width, cords.y + length));

        noise_height += value * amplitude;
        amplitude *= persistance;
        freq *= lacunatrity;

    }
    return noise_height;
}

//I think the parameters could be narrowed by creativly using the settings struct. 
//If I had more time, I would refactor it since this is a bit of a pile of variables
grid generate_grid(double unit_w, double unit_l, int s_width, int s_length,
                     vec3* pos, int3* tri_indices, vec3* colors, 
                     double offset_w, double offset_l, double offset_h,
                     double height_scale, vec3* palette, int num_colors, 
                     int octaves, double persistance, double lacunarity, int mode) {
    grid new_grid;
    int count = new_grid.count; int tri_count = new_grid.tri_count;
    new_grid.length = s_length; new_grid.width = s_width;
    new_grid.offset_w = offset_w; new_grid.offset_l = offset_l;
    double max = -HUGE; double min = HUGE;
    new_grid.octaves = octaves; new_grid.persistance = persistance; new_grid.lacunarity = lacunarity;
    new_grid.mode = mode;
    double width = unit_w * s_width; double length = unit_l * s_length;
    for (int i = 0; i < s_width; i++){
        for (int j = 0; j < s_length; j++){
            double f = .5;               
            double h1 = octave_noise(f*V2(2*j*unit_l + offset_l, 2*i*unit_w + offset_w), octaves, persistance, lacunarity, mode, width, length);                   if(h1 > max) max = h1; if(h1 < min) min = h1;
            double h2 = octave_noise(f*V2(2*j*unit_l + unit_l + offset_l, 2*i*unit_w + offset_w), octaves, persistance, lacunarity, mode, width, length);          if(h2 > max) max = h2; if(h2 < min) min = h2;  
            double h3 = octave_noise(f*V2(2*j*unit_l + offset_l, 2*i*unit_l  + unit_w + offset_w), octaves, persistance, lacunarity, mode, width, length);         if(h3 > max) max = h3; if(h3 < min) min = h3;    
            double h4 = octave_noise(f*V2(2*j*unit_l + unit_l + offset_l, 2*i*unit_w + unit_w + offset_w), octaves, persistance, lacunarity, mode, width, length); if(h4 > max) max = h4; if(h4 < min) min = h4;
            pos[count] =     V3(2*j*unit_l,          height_scale * h1 + offset_h, 2*i*unit_w); 
            pos[count + 1] = V3(2*j*unit_l + unit_l, height_scale * h2 + offset_h, 2*i*unit_w); 
            pos[count + 2] = V3(2*j*unit_l,          height_scale * h3 + offset_h, 2*i*unit_w + unit_w); 
            pos[count + 3] = V3(2*j*unit_l + unit_l, height_scale * h4 + offset_h, 2*i*unit_w + unit_w); 
            
            //if triangles can be placed behind, place them behind
            if (j != 0){
                tri_indices[tri_count] = {count - 3, count, count + 2};
                tri_indices[tri_count + 1] = {count - 3, count + 2, count - 1};
                tri_count += 2;
                //if can be places over, too
                if (i != 0){
                    tri_indices[tri_count] = {count - (4*s_length) - 1, count - 3, count};
                    tri_indices[tri_count + 1] = {count - (4*s_length) - 1, count - (4*s_length) + 2, count};
                    tri_count += 2;
                }
            } 
            //place tri infrount
            tri_indices[tri_count] = {count, count + 1, count + 3};
            tri_indices[tri_count + 1] = {count, count + 3, count + 2};
            tri_count += 2;
            //if trangles can be placed over, place over
            if (i != 0){
                tri_indices[tri_count] = {count - (4*s_length) + 2, count - (4*s_length) + 3, count + 1};
                tri_indices[tri_count + 1] = {count - (4*s_length) + 2, count + 1, count};
                tri_count += 2;
            }
            count += 4;
        }
    }
    //normalizing heights
    double amp = (max - min);
    count = 0;
    for (int i = 0; i < s_width; i++){
        for (int j = 0; j < s_length; j++){
            for (int k = 0; k < 4; k++){
                colors[count] = color_gen(pos[count].y, palette, num_colors, height_scale*max, height_scale*min);
                pos[count].y *= (2/amp);
                
                count ++;
            }
        }
    } 
    new_grid.colors = colors;
    new_grid.tri_indices = tri_indices;
    new_grid.pos = pos;
    new_grid.count = count;
    new_grid.tri_count = tri_count;
    new_grid.height_scale = height_scale;
    return new_grid;
}

struct settings{
    int length_density;
    int width_density;
    double offset_x;
    double offset_y;
    double height_scale;
    int water_octaves;
    int land_octaves;
    double persistance;
    double lacunarity;
    int type_of_noise;
    bool playing;
    bool show_points;
};

//a way of generating randomized set of settings at runtime
//todo in future - use normal distibution to constrain the extent of the vars instead of hard coding - make more natural gen
settings generate_reasonable_settings(){
    settings s;
    srand(time(NULL)); 
    s.length_density = (int)((rand() % 40) + 20);
    s.width_density = s.length_density;
    s.type_of_noise = (int)((rand() % 3) + 1);
    s.offset_x = s.type_of_noise = 1 ? (rand() % 100) : (rand() % 10) - 5;
    s.offset_y = s.type_of_noise = 1 ? (rand() % 100) : (rand() % 10) - 5;
    s.height_scale = (rand() % 3) + 0.75;
    s.water_octaves = (int)((rand() % 5) + 1);
    s.land_octaves = (int)((rand() % 5) + 1);
    s.persistance = ((double)(rand() % 5) / 10.) + 0.2;
    s.lacunarity = ((double)(rand() % 300) / 100.) + 1;
    s.playing = true;
    s.show_points = false;
    return s;
}

//all of the code to actully call the fancy draw functions
void draw() {
    settings s = generate_reasonable_settings();
    //side length in world cords
    double real_width = 5;
    double real_length = 5;
    //how many pairs of triangles on each side
    int max_size = 100;
    int s_width = s.width_density;
    int s_length = s.length_density; 
    //offsets 
    double max_offset = 100;
    double offset_w = s.offset_x;
    double offset_l = s.offset_y;
    double w_offset_w = 10.0;
    double w_offset_l = 0.0;
    double land_height_scale = s.height_scale;
    double max_height_scale = 10;
    //how many triangles per side
    //side length of a triangle in world cords
    double unit_w = real_width / s_width;
    double unit_l = real_length / s_length;
    
    int num_triangles = max_size * max_size * 8;
    int num_points = (1+2*max_size) * (1+2*max_size);

    //octave settings
    int mode = 2; // 1 = simplex, 2 = perlin
    
    int octaves = s.land_octaves;
    int w_octaves = s.water_octaves;
    double persistance = s.persistance;
    double lacunarity = s.lacunarity;
    
    vec3* pos = (vec3*)malloc(num_points * sizeof(vec3));
    vec3* water_pos = (vec3*)malloc(num_points * sizeof(vec3));
    vec3* colors = (vec3*)malloc(num_points * sizeof(vec3));
    vec3* water_colors = (vec3*)malloc(num_points * sizeof(vec3));
    int3* tri_indices = (int3*)malloc(num_triangles * sizeof(int3));
    int3* water_tri_indices = (int3*)malloc(num_triangles * sizeof(int3));

    //set color palettes
    int num_world_colors = 5;
    vec3* world_cols = (vec3*)malloc(num_world_colors*sizeof(vec3));
    world_cols[0] = V3(76,172,188) / 255; // blue
    world_cols[1] = V3(108,196,161) / 255; // aqua
    world_cols[2] = V3(160,217,149) / 255; // green 
    world_cols[3] = V3(246,227,197) / 255; // tan
    world_cols[4] = V3(188,188,188) / 255; // grey
    int num_water_colors = 4;
    vec3* water_cols = (vec3*)malloc(num_water_colors*sizeof(vec3));
    water_cols[0] = V3(13,76,146) / 255; // dark blue
    water_cols[1] = V3(89,193,189) / 255; // aqua
    water_cols[2] = V3(160,228,203) / 255; // green blue
    water_cols[3] = V3(207,245,231) / 255; // light blue

    //init the meshes
    grid new_grid = generate_grid(unit_w, unit_l, s_width, s_length, pos, tri_indices, colors, offset_w, offset_l, 0, land_height_scale, world_cols, num_world_colors, octaves, persistance, lacunarity, mode);
    grid water = generate_grid(unit_w, unit_l, s_width, s_length, water_pos, water_tri_indices, water_colors, offset_w, offset_l, 0.1, 0.3, water_cols, num_water_colors, 1, 0.5, 2, mode); 

    Camera3D camera = { 20, RAD(45), RAD(0), RAD(-10), real_length / 2, 0.5 };

    //ui settings
    bool ui = true;
    bool playing = true;
    bool show = false;
    double t = 0;

    //general loop
    while(begin_frame()){
        
        camera_move(&camera);
        mat4 P = camera_get_P(&camera); mat4 V = camera_get_V(&camera);
        mat4 PV = camera_get_PV(&camera);

        //skybox
        //pngs from https://www.cleanpng.com/png-spacecape-deviantart-digital-art-symbol-night-sky-1023556/
        glDisable(GL_DEPTH_TEST); {
            int3 triangle_indices[] = { { 0, 1, 2 }, { 0, 2, 3} };
            vec2 vertex_texCoords[] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };

            mat4 M = Identity4x4;
            M *= Translation(camera_get_origin(&camera));
              
            vec3 back_vertex_positions[] = {
                { -1, -1, -1 },
                {  1, -1, -1 },
                {  1,  1, -1 },
                { -1,  1, -1 },
            };
            fancy_draw(P, V, M,
                    2, triangle_indices, 4, back_vertex_positions,
                    NULL, NULL, {},
                    vertex_texCoords, "new_back.png");
            vec3 front_vertex_positions[] = {
                {  1, -1, 1 },
                { -1, -1, 1 },
                { -1,  1, 1 },
                {  1,  1, 1 },
            };
            fancy_draw(P, V, M,
                    2, triangle_indices, 4, front_vertex_positions,
                    NULL, NULL, {},
                    vertex_texCoords, "new_front.png");
            vec3 left_vertex_positions[] = {
                { -1, -1,  1 },
                { -1, -1, -1 },
                { -1,  1, -1 },
                { -1,  1,  1 },
            };
            fancy_draw(P, V, M,
                    2, triangle_indices, 4, left_vertex_positions,
                    NULL, NULL, {},
                    vertex_texCoords, "new_left.png");
            vec3 right_vertex_positions[] = {
                { 1, -1, -1 },
                { 1, -1,  1 },
                { 1,  1,  1 },
                { 1,  1, -1 },
            };
            fancy_draw(P, V, M,
                    2, triangle_indices, 4, right_vertex_positions,
                    NULL, NULL, {},
                    vertex_texCoords, "new_right.png");
            vec3 bottom_vertex_positions[] = {
                {  -1, -1, 1 },
                { 1, -1,  1 },
                {  1, -1, -1 },
                {  -1, -1, -1 },
            };
            fancy_draw(P, V, M,
                    2, triangle_indices, 4, bottom_vertex_positions,
                    NULL, NULL, {},
                    vertex_texCoords, "new_bottom.png");
            vec3 top_vertex_positions[] = {
                {  -1,  1, -1 },
                {  1,  1,  -1 },
                { 1,  1,  1 },
                { -1,  1, 1 },
            };
            fancy_draw(P, V, M,
                    2, triangle_indices, 4, top_vertex_positions,
                    NULL, NULL, {},
                    vertex_texCoords, "new_top.png");
            // TODO draw the front, left, right, bottom, and top

        } glEnable(GL_DEPTH_TEST);

        //ui settings
        imgui_checkbox("Show UI", &ui, 'u');
        if(ui){
            if (imgui_button("New Map", 'n')) {
                s = generate_reasonable_settings();
                //reset all the map vars
                s_width = s.width_density;
                s_length = s.length_density; 
                offset_w = s.offset_x;
                offset_l = s.offset_y;
                land_height_scale = s.height_scale;
                octaves = s.land_octaves;
                w_octaves = s.water_octaves;
                persistance = s.persistance;
                lacunarity = s.lacunarity;
            }
            imgui_slider("Length Density", &s_length, 0, max_size);
            imgui_slider("Width Density", &s_width, 0, max_size);
            
            if(mode == 1){
                imgui_slider("Offset X", &offset_l, -max_offset, max_offset);
                imgui_slider("Offset Y", &offset_w, -max_offset, max_offset);
            }
            else {  
                imgui_slider("Offset X", &offset_l, 0, max_offset);
                imgui_slider("Offset Y", &offset_w, 0, max_offset);
            }
            
            imgui_slider("Height Scale", &land_height_scale, 0.1, max_height_scale);
            imgui_slider("Water Octaves", &w_octaves, 1, 10);
            imgui_slider("Land Octaves", &octaves, 1, 10);
            imgui_slider("Persistance", &persistance, 0.1, 1);
            imgui_slider("Lacunarity", &lacunarity, 1, 10);
            imgui_checkbox("Playing", &playing, 'p');
            imgui_checkbox("Show Points", &show, 's');
            if(mode == 1) imgui_slider("Simplex Noise", &mode, 1, 3);
            if(mode == 2) imgui_slider("Perlin Noise", &mode, 1, 3);
            if(mode == 3) imgui_slider("Periodic Perlin Noise", &mode, 1, 3);
        }
        //adjustable resolution
        if(s_length != new_grid.length || s_width != new_grid.width){
            //update vars
            unit_w = real_width / s_width; unit_l = real_length / s_length;
            num_triangles = s_length * s_width * 8;
            num_points = (1+2*s_length) * (1+2*s_width);
            //redraw
            new_grid = generate_grid(unit_w, unit_l, s_width, s_length, pos, tri_indices, colors, offset_w, offset_l, 0, land_height_scale, world_cols, num_world_colors, octaves, persistance, lacunarity, mode);
        }
        //other adjustments
        if(offset_w != new_grid.offset_w || offset_l != new_grid.offset_l || land_height_scale != new_grid.height_scale || octaves != new_grid.octaves || persistance != new_grid.persistance || lacunarity != new_grid.lacunarity || mode != new_grid.mode){
            //redraw
            if(mode != new_grid.mode && mode == 2) offset_l = 10;
            if(mode != new_grid.mode && mode == 1) offset_l = -10;
            new_grid = generate_grid(unit_w, unit_l, s_width, s_length, pos, tri_indices, colors, offset_w, offset_l, 0, land_height_scale, world_cols, num_world_colors, octaves, persistance, lacunarity, mode);
        }

        //animating water
        if(playing) {
            double offset = t;
            if (mode == 1) {
                offset = sin(t); 
                w_offset_w = -10;
            }
            else w_offset_w = 10;
            //not quite working with simplex
            water = generate_grid(unit_w, unit_l, s_width, s_length, water_pos, water_tri_indices, water_colors, w_offset_w + t, w_offset_l, 0.1, 0.3, water_cols, num_water_colors,  4, 0.5, 2, 2);
            t += .0167;
        }

        if(show) basic_draw(POINTS, PV, num_points, pos, monokai.white);

        //land mesh
        fancy_draw(P, V, Identity4x4, new_grid.tri_count, new_grid.tri_indices, new_grid.count, new_grid.pos, NULL, new_grid.colors);

        //water mesh (hack_a changes the transperency)
        hack_a = .7;
        fancy_draw(P, V, Identity4x4, water.tri_count, water.tri_indices, water.count, water.pos, NULL, water.colors); //V4(*water.colors, 0.3)
        hack_a = 1.;
    }
    //free all malloced vars
    free(pos); free(water_pos);
    free(colors); free(water_colors);
    free(tri_indices); free(water_tri_indices);
}

int main() {
    init();
    draw();
    return 0;
}