// 
// main.c: Demo of many target regions in different source modules
// 

#include <stdio.h>
#include <stdlib.h>
#include "myfns.h"

int main(){
   const int N = 100000;    
   int ncases = 1000;
   int a[N],b[N],p[N],pcheck[N],s[N],scheck[N];
   int flag=-1;
   for(int i=0;i<N;i++) {
      a[i]=i+1;
      b[i]=i+2;
      pcheck[i]=a[i]*b[i];
      scheck[i]=a[i]+b[i];
   }

   int r = rand() ;
   int selector = r % ncases;
   printf("selector %d\n",selector);
 
   switch(selector) {
     case 0:
       vmul_000(a,b,p,N);
       vsum_000(a,b,s,N);
       break;
     case 1:
       vmul_001(a,b,p,N);
       vsum_001(a,b,s,N);
       break;
     case 2:
       vmul_002(a,b,p,N);
       vsum_002(a,b,s,N);
       break;
     case 3:
       vmul_003(a,b,p,N);
       vsum_003(a,b,s,N);
       break;
     case 4:
       vmul_004(a,b,p,N);
       vsum_004(a,b,s,N);
       break;
     case 5:
       vmul_005(a,b,p,N);
       vsum_005(a,b,s,N);
       break;
     case 6:
       vmul_006(a,b,p,N);
       vsum_006(a,b,s,N);
       break;
     case 7:
       vmul_007(a,b,p,N);
       vsum_007(a,b,s,N);
       break;
     case 8:
       vmul_008(a,b,p,N);
       vsum_008(a,b,s,N);
       break;
     case 9:
       vmul_009(a,b,p,N);
       vsum_009(a,b,s,N);
       break;
     case 10:
       vmul_010(a,b,p,N);
       vsum_010(a,b,s,N);
       break;
     case 11:
       vmul_011(a,b,p,N);
       vsum_011(a,b,s,N);
       break;
     case 12:
       vmul_012(a,b,p,N);
       vsum_012(a,b,s,N);
       break;
     case 13:
       vmul_013(a,b,p,N);
       vsum_013(a,b,s,N);
       break;
     case 14:
       vmul_014(a,b,p,N);
       vsum_014(a,b,s,N);
       break;
     case 15:
       vmul_015(a,b,p,N);
       vsum_015(a,b,s,N);
       break;
     case 16:
       vmul_016(a,b,p,N);
       vsum_016(a,b,s,N);
       break;
     case 17:
       vmul_017(a,b,p,N);
       vsum_017(a,b,s,N);
       break;
     case 18:
       vmul_018(a,b,p,N);
       vsum_018(a,b,s,N);
       break;
     case 19:
       vmul_019(a,b,p,N);
       vsum_019(a,b,s,N);
       break;
     case 20:
       vmul_020(a,b,p,N);
       vsum_020(a,b,s,N);
       break;
     case 21:
       vmul_021(a,b,p,N);
       vsum_021(a,b,s,N);
       break;
     case 22:
       vmul_022(a,b,p,N);
       vsum_022(a,b,s,N);
       break;
     case 23:
       vmul_023(a,b,p,N);
       vsum_023(a,b,s,N);
       break;
     case 24:
       vmul_024(a,b,p,N);
       vsum_024(a,b,s,N);
       break;
     case 25:
       vmul_025(a,b,p,N);
       vsum_025(a,b,s,N);
       break;
     case 26:
       vmul_026(a,b,p,N);
       vsum_026(a,b,s,N);
       break;
     case 27:
       vmul_027(a,b,p,N);
       vsum_027(a,b,s,N);
       break;
     case 28:
       vmul_028(a,b,p,N);
       vsum_028(a,b,s,N);
       break;
     case 29:
       vmul_029(a,b,p,N);
       vsum_029(a,b,s,N);
       break;
     case 30:
       vmul_030(a,b,p,N);
       vsum_030(a,b,s,N);
       break;
     case 31:
       vmul_031(a,b,p,N);
       vsum_031(a,b,s,N);
       break;
     case 32:
       vmul_032(a,b,p,N);
       vsum_032(a,b,s,N);
       break;
     case 33:
       vmul_033(a,b,p,N);
       vsum_033(a,b,s,N);
       break;
     case 34:
       vmul_034(a,b,p,N);
       vsum_034(a,b,s,N);
       break;
     case 35:
       vmul_035(a,b,p,N);
       vsum_035(a,b,s,N);
       break;
     case 36:
       vmul_036(a,b,p,N);
       vsum_036(a,b,s,N);
       break;
     case 37:
       vmul_037(a,b,p,N);
       vsum_037(a,b,s,N);
       break;
     case 38:
       vmul_038(a,b,p,N);
       vsum_038(a,b,s,N);
       break;
     case 39:
       vmul_039(a,b,p,N);
       vsum_039(a,b,s,N);
       break;
     case 40:
       vmul_040(a,b,p,N);
       vsum_040(a,b,s,N);
       break;
     case 41:
       vmul_041(a,b,p,N);
       vsum_041(a,b,s,N);
       break;
     case 42:
       vmul_042(a,b,p,N);
       vsum_042(a,b,s,N);
       break;
     case 43:
       vmul_043(a,b,p,N);
       vsum_043(a,b,s,N);
       break;
     case 44:
       vmul_044(a,b,p,N);
       vsum_044(a,b,s,N);
       break;
     case 45:
       vmul_045(a,b,p,N);
       vsum_045(a,b,s,N);
       break;
     case 46:
       vmul_046(a,b,p,N);
       vsum_046(a,b,s,N);
       break;
     case 47:
       vmul_047(a,b,p,N);
       vsum_047(a,b,s,N);
       break;
     case 48:
       vmul_048(a,b,p,N);
       vsum_048(a,b,s,N);
       break;
     case 49:
       vmul_049(a,b,p,N);
       vsum_049(a,b,s,N);
       break;
     case 50:
       vmul_050(a,b,p,N);
       vsum_050(a,b,s,N);
       break;
     case 51:
       vmul_051(a,b,p,N);
       vsum_051(a,b,s,N);
       break;
     case 52:
       vmul_052(a,b,p,N);
       vsum_052(a,b,s,N);
       break;
     case 53:
       vmul_053(a,b,p,N);
       vsum_053(a,b,s,N);
       break;
     case 54:
       vmul_054(a,b,p,N);
       vsum_054(a,b,s,N);
       break;
     case 55:
       vmul_055(a,b,p,N);
       vsum_055(a,b,s,N);
       break;
     case 56:
       vmul_056(a,b,p,N);
       vsum_056(a,b,s,N);
       break;
     case 57:
       vmul_057(a,b,p,N);
       vsum_057(a,b,s,N);
       break;
     case 58:
       vmul_058(a,b,p,N);
       vsum_058(a,b,s,N);
       break;
     case 59:
       vmul_059(a,b,p,N);
       vsum_059(a,b,s,N);
       break;
     case 60:
       vmul_060(a,b,p,N);
       vsum_060(a,b,s,N);
       break;
     case 61:
       vmul_061(a,b,p,N);
       vsum_061(a,b,s,N);
       break;
     case 62:
       vmul_062(a,b,p,N);
       vsum_062(a,b,s,N);
       break;
     case 63:
       vmul_063(a,b,p,N);
       vsum_063(a,b,s,N);
       break;
     case 64:
       vmul_064(a,b,p,N);
       vsum_064(a,b,s,N);
       break;
     case 65:
       vmul_065(a,b,p,N);
       vsum_065(a,b,s,N);
       break;
     case 66:
       vmul_066(a,b,p,N);
       vsum_066(a,b,s,N);
       break;
     case 67:
       vmul_067(a,b,p,N);
       vsum_067(a,b,s,N);
       break;
     case 68:
       vmul_068(a,b,p,N);
       vsum_068(a,b,s,N);
       break;
     case 69:
       vmul_069(a,b,p,N);
       vsum_069(a,b,s,N);
       break;
     case 70:
       vmul_070(a,b,p,N);
       vsum_070(a,b,s,N);
       break;
     case 71:
       vmul_071(a,b,p,N);
       vsum_071(a,b,s,N);
       break;
     case 72:
       vmul_072(a,b,p,N);
       vsum_072(a,b,s,N);
       break;
     case 73:
       vmul_073(a,b,p,N);
       vsum_073(a,b,s,N);
       break;
     case 74:
       vmul_074(a,b,p,N);
       vsum_074(a,b,s,N);
       break;
     case 75:
       vmul_075(a,b,p,N);
       vsum_075(a,b,s,N);
       break;
     case 76:
       vmul_076(a,b,p,N);
       vsum_076(a,b,s,N);
       break;
     case 77:
       vmul_077(a,b,p,N);
       vsum_077(a,b,s,N);
       break;
     case 78:
       vmul_078(a,b,p,N);
       vsum_078(a,b,s,N);
       break;
     case 79:
       vmul_079(a,b,p,N);
       vsum_079(a,b,s,N);
       break;
     case 80:
       vmul_080(a,b,p,N);
       vsum_080(a,b,s,N);
       break;
     case 81:
       vmul_081(a,b,p,N);
       vsum_081(a,b,s,N);
       break;
     case 82:
       vmul_082(a,b,p,N);
       vsum_082(a,b,s,N);
       break;
     case 83:
       vmul_083(a,b,p,N);
       vsum_083(a,b,s,N);
       break;
     case 84:
       vmul_084(a,b,p,N);
       vsum_084(a,b,s,N);
       break;
     case 85:
       vmul_085(a,b,p,N);
       vsum_085(a,b,s,N);
       break;
     case 86:
       vmul_086(a,b,p,N);
       vsum_086(a,b,s,N);
       break;
     case 87:
       vmul_087(a,b,p,N);
       vsum_087(a,b,s,N);
       break;
     case 88:
       vmul_088(a,b,p,N);
       vsum_088(a,b,s,N);
       break;
     case 89:
       vmul_089(a,b,p,N);
       vsum_089(a,b,s,N);
       break;
     case 90:
       vmul_090(a,b,p,N);
       vsum_090(a,b,s,N);
       break;
     case 91:
       vmul_091(a,b,p,N);
       vsum_091(a,b,s,N);
       break;
     case 92:
       vmul_092(a,b,p,N);
       vsum_092(a,b,s,N);
       break;
     case 93:
       vmul_093(a,b,p,N);
       vsum_093(a,b,s,N);
       break;
     case 94:
       vmul_094(a,b,p,N);
       vsum_094(a,b,s,N);
       break;
     case 95:
       vmul_095(a,b,p,N);
       vsum_095(a,b,s,N);
       break;
     case 96:
       vmul_096(a,b,p,N);
       vsum_096(a,b,s,N);
       break;
     case 97:
       vmul_097(a,b,p,N);
       vsum_097(a,b,s,N);
       break;
     case 98:
       vmul_098(a,b,p,N);
       vsum_098(a,b,s,N);
       break;
     case 99:
       vmul_099(a,b,p,N);
       vsum_099(a,b,s,N);
       break;
     case 100:
       vmul_100(a,b,p,N);
       vsum_100(a,b,s,N);
       break;
     case 101:
       vmul_101(a,b,p,N);
       vsum_101(a,b,s,N);
       break;
     case 102:
       vmul_102(a,b,p,N);
       vsum_102(a,b,s,N);
       break;
     case 103:
       vmul_103(a,b,p,N);
       vsum_103(a,b,s,N);
       break;
     case 104:
       vmul_104(a,b,p,N);
       vsum_104(a,b,s,N);
       break;
     case 105:
       vmul_105(a,b,p,N);
       vsum_105(a,b,s,N);
       break;
     case 106:
       vmul_106(a,b,p,N);
       vsum_106(a,b,s,N);
       break;
     case 107:
       vmul_107(a,b,p,N);
       vsum_107(a,b,s,N);
       break;
     case 108:
       vmul_108(a,b,p,N);
       vsum_108(a,b,s,N);
       break;
     case 109:
       vmul_109(a,b,p,N);
       vsum_109(a,b,s,N);
       break;
     case 110:
       vmul_110(a,b,p,N);
       vsum_110(a,b,s,N);
       break;
     case 111:
       vmul_111(a,b,p,N);
       vsum_111(a,b,s,N);
       break;
     case 112:
       vmul_112(a,b,p,N);
       vsum_112(a,b,s,N);
       break;
     case 113:
       vmul_113(a,b,p,N);
       vsum_113(a,b,s,N);
       break;
     case 114:
       vmul_114(a,b,p,N);
       vsum_114(a,b,s,N);
       break;
     case 115:
       vmul_115(a,b,p,N);
       vsum_115(a,b,s,N);
       break;
     case 116:
       vmul_116(a,b,p,N);
       vsum_116(a,b,s,N);
       break;
     case 117:
       vmul_117(a,b,p,N);
       vsum_117(a,b,s,N);
       break;
     case 118:
       vmul_118(a,b,p,N);
       vsum_118(a,b,s,N);
       break;
     case 119:
       vmul_119(a,b,p,N);
       vsum_119(a,b,s,N);
       break;
     case 120:
       vmul_120(a,b,p,N);
       vsum_120(a,b,s,N);
       break;
     case 121:
       vmul_121(a,b,p,N);
       vsum_121(a,b,s,N);
       break;
     case 122:
       vmul_122(a,b,p,N);
       vsum_122(a,b,s,N);
       break;
     case 123:
       vmul_123(a,b,p,N);
       vsum_123(a,b,s,N);
       break;
     case 124:
       vmul_124(a,b,p,N);
       vsum_124(a,b,s,N);
       break;
     case 125:
       vmul_125(a,b,p,N);
       vsum_125(a,b,s,N);
       break;
     case 126:
       vmul_126(a,b,p,N);
       vsum_126(a,b,s,N);
       break;
     case 127:
       vmul_127(a,b,p,N);
       vsum_127(a,b,s,N);
       break;
     case 128:
       vmul_128(a,b,p,N);
       vsum_128(a,b,s,N);
       break;
     case 129:
       vmul_129(a,b,p,N);
       vsum_129(a,b,s,N);
       break;
     case 130:
       vmul_130(a,b,p,N);
       vsum_130(a,b,s,N);
       break;
     case 131:
       vmul_131(a,b,p,N);
       vsum_131(a,b,s,N);
       break;
     case 132:
       vmul_132(a,b,p,N);
       vsum_132(a,b,s,N);
       break;
     case 133:
       vmul_133(a,b,p,N);
       vsum_133(a,b,s,N);
       break;
     case 134:
       vmul_134(a,b,p,N);
       vsum_134(a,b,s,N);
       break;
     case 135:
       vmul_135(a,b,p,N);
       vsum_135(a,b,s,N);
       break;
     case 136:
       vmul_136(a,b,p,N);
       vsum_136(a,b,s,N);
       break;
     case 137:
       vmul_137(a,b,p,N);
       vsum_137(a,b,s,N);
       break;
     case 138:
       vmul_138(a,b,p,N);
       vsum_138(a,b,s,N);
       break;
     case 139:
       vmul_139(a,b,p,N);
       vsum_139(a,b,s,N);
       break;
     case 140:
       vmul_140(a,b,p,N);
       vsum_140(a,b,s,N);
       break;
     case 141:
       vmul_141(a,b,p,N);
       vsum_141(a,b,s,N);
       break;
     case 142:
       vmul_142(a,b,p,N);
       vsum_142(a,b,s,N);
       break;
     case 143:
       vmul_143(a,b,p,N);
       vsum_143(a,b,s,N);
       break;
     case 144:
       vmul_144(a,b,p,N);
       vsum_144(a,b,s,N);
       break;
     case 145:
       vmul_145(a,b,p,N);
       vsum_145(a,b,s,N);
       break;
     case 146:
       vmul_146(a,b,p,N);
       vsum_146(a,b,s,N);
       break;
     case 147:
       vmul_147(a,b,p,N);
       vsum_147(a,b,s,N);
       break;
     case 148:
       vmul_148(a,b,p,N);
       vsum_148(a,b,s,N);
       break;
     case 149:
       vmul_149(a,b,p,N);
       vsum_149(a,b,s,N);
       break;
     case 150:
       vmul_150(a,b,p,N);
       vsum_150(a,b,s,N);
       break;
     case 151:
       vmul_151(a,b,p,N);
       vsum_151(a,b,s,N);
       break;
     case 152:
       vmul_152(a,b,p,N);
       vsum_152(a,b,s,N);
       break;
     case 153:
       vmul_153(a,b,p,N);
       vsum_153(a,b,s,N);
       break;
     case 154:
       vmul_154(a,b,p,N);
       vsum_154(a,b,s,N);
       break;
     case 155:
       vmul_155(a,b,p,N);
       vsum_155(a,b,s,N);
       break;
     case 156:
       vmul_156(a,b,p,N);
       vsum_156(a,b,s,N);
       break;
     case 157:
       vmul_157(a,b,p,N);
       vsum_157(a,b,s,N);
       break;
     case 158:
       vmul_158(a,b,p,N);
       vsum_158(a,b,s,N);
       break;
     case 159:
       vmul_159(a,b,p,N);
       vsum_159(a,b,s,N);
       break;
     case 160:
       vmul_160(a,b,p,N);
       vsum_160(a,b,s,N);
       break;
     case 161:
       vmul_161(a,b,p,N);
       vsum_161(a,b,s,N);
       break;
     case 162:
       vmul_162(a,b,p,N);
       vsum_162(a,b,s,N);
       break;
     case 163:
       vmul_163(a,b,p,N);
       vsum_163(a,b,s,N);
       break;
     case 164:
       vmul_164(a,b,p,N);
       vsum_164(a,b,s,N);
       break;
     case 165:
       vmul_165(a,b,p,N);
       vsum_165(a,b,s,N);
       break;
     case 166:
       vmul_166(a,b,p,N);
       vsum_166(a,b,s,N);
       break;
     case 167:
       vmul_167(a,b,p,N);
       vsum_167(a,b,s,N);
       break;
     case 168:
       vmul_168(a,b,p,N);
       vsum_168(a,b,s,N);
       break;
     case 169:
       vmul_169(a,b,p,N);
       vsum_169(a,b,s,N);
       break;
     case 170:
       vmul_170(a,b,p,N);
       vsum_170(a,b,s,N);
       break;
     case 171:
       vmul_171(a,b,p,N);
       vsum_171(a,b,s,N);
       break;
     case 172:
       vmul_172(a,b,p,N);
       vsum_172(a,b,s,N);
       break;
     case 173:
       vmul_173(a,b,p,N);
       vsum_173(a,b,s,N);
       break;
     case 174:
       vmul_174(a,b,p,N);
       vsum_174(a,b,s,N);
       break;
     case 175:
       vmul_175(a,b,p,N);
       vsum_175(a,b,s,N);
       break;
     case 176:
       vmul_176(a,b,p,N);
       vsum_176(a,b,s,N);
       break;
     case 177:
       vmul_177(a,b,p,N);
       vsum_177(a,b,s,N);
       break;
     case 178:
       vmul_178(a,b,p,N);
       vsum_178(a,b,s,N);
       break;
     case 179:
       vmul_179(a,b,p,N);
       vsum_179(a,b,s,N);
       break;
     case 180:
       vmul_180(a,b,p,N);
       vsum_180(a,b,s,N);
       break;
     case 181:
       vmul_181(a,b,p,N);
       vsum_181(a,b,s,N);
       break;
     case 182:
       vmul_182(a,b,p,N);
       vsum_182(a,b,s,N);
       break;
     case 183:
       vmul_183(a,b,p,N);
       vsum_183(a,b,s,N);
       break;
     case 184:
       vmul_184(a,b,p,N);
       vsum_184(a,b,s,N);
       break;
     case 185:
       vmul_185(a,b,p,N);
       vsum_185(a,b,s,N);
       break;
     case 186:
       vmul_186(a,b,p,N);
       vsum_186(a,b,s,N);
       break;
     case 187:
       vmul_187(a,b,p,N);
       vsum_187(a,b,s,N);
       break;
     case 188:
       vmul_188(a,b,p,N);
       vsum_188(a,b,s,N);
       break;
     case 189:
       vmul_189(a,b,p,N);
       vsum_189(a,b,s,N);
       break;
     case 190:
       vmul_190(a,b,p,N);
       vsum_190(a,b,s,N);
       break;
     case 191:
       vmul_191(a,b,p,N);
       vsum_191(a,b,s,N);
       break;
     case 192:
       vmul_192(a,b,p,N);
       vsum_192(a,b,s,N);
       break;
     case 193:
       vmul_193(a,b,p,N);
       vsum_193(a,b,s,N);
       break;
     case 194:
       vmul_194(a,b,p,N);
       vsum_194(a,b,s,N);
       break;
     case 195:
       vmul_195(a,b,p,N);
       vsum_195(a,b,s,N);
       break;
     case 196:
       vmul_196(a,b,p,N);
       vsum_196(a,b,s,N);
       break;
     case 197:
       vmul_197(a,b,p,N);
       vsum_197(a,b,s,N);
       break;
     case 198:
       vmul_198(a,b,p,N);
       vsum_198(a,b,s,N);
       break;
     case 199:
       vmul_199(a,b,p,N);
       vsum_199(a,b,s,N);
       break;
     case 200:
       vmul_200(a,b,p,N);
       vsum_200(a,b,s,N);
       break;
     case 201:
       vmul_201(a,b,p,N);
       vsum_201(a,b,s,N);
       break;
     case 202:
       vmul_202(a,b,p,N);
       vsum_202(a,b,s,N);
       break;
     case 203:
       vmul_203(a,b,p,N);
       vsum_203(a,b,s,N);
       break;
     case 204:
       vmul_204(a,b,p,N);
       vsum_204(a,b,s,N);
       break;
     case 205:
       vmul_205(a,b,p,N);
       vsum_205(a,b,s,N);
       break;
     case 206:
       vmul_206(a,b,p,N);
       vsum_206(a,b,s,N);
       break;
     case 207:
       vmul_207(a,b,p,N);
       vsum_207(a,b,s,N);
       break;
     case 208:
       vmul_208(a,b,p,N);
       vsum_208(a,b,s,N);
       break;
     case 209:
       vmul_209(a,b,p,N);
       vsum_209(a,b,s,N);
       break;
     case 210:
       vmul_210(a,b,p,N);
       vsum_210(a,b,s,N);
       break;
     case 211:
       vmul_211(a,b,p,N);
       vsum_211(a,b,s,N);
       break;
     case 212:
       vmul_212(a,b,p,N);
       vsum_212(a,b,s,N);
       break;
     case 213:
       vmul_213(a,b,p,N);
       vsum_213(a,b,s,N);
       break;
     case 214:
       vmul_214(a,b,p,N);
       vsum_214(a,b,s,N);
       break;
     case 215:
       vmul_215(a,b,p,N);
       vsum_215(a,b,s,N);
       break;
     case 216:
       vmul_216(a,b,p,N);
       vsum_216(a,b,s,N);
       break;
     case 217:
       vmul_217(a,b,p,N);
       vsum_217(a,b,s,N);
       break;
     case 218:
       vmul_218(a,b,p,N);
       vsum_218(a,b,s,N);
       break;
     case 219:
       vmul_219(a,b,p,N);
       vsum_219(a,b,s,N);
       break;
     case 220:
       vmul_220(a,b,p,N);
       vsum_220(a,b,s,N);
       break;
     case 221:
       vmul_221(a,b,p,N);
       vsum_221(a,b,s,N);
       break;
     case 222:
       vmul_222(a,b,p,N);
       vsum_222(a,b,s,N);
       break;
     case 223:
       vmul_223(a,b,p,N);
       vsum_223(a,b,s,N);
       break;
     case 224:
       vmul_224(a,b,p,N);
       vsum_224(a,b,s,N);
       break;
     case 225:
       vmul_225(a,b,p,N);
       vsum_225(a,b,s,N);
       break;
     case 226:
       vmul_226(a,b,p,N);
       vsum_226(a,b,s,N);
       break;
     case 227:
       vmul_227(a,b,p,N);
       vsum_227(a,b,s,N);
       break;
     case 228:
       vmul_228(a,b,p,N);
       vsum_228(a,b,s,N);
       break;
     case 229:
       vmul_229(a,b,p,N);
       vsum_229(a,b,s,N);
       break;
     case 230:
       vmul_230(a,b,p,N);
       vsum_230(a,b,s,N);
       break;
     case 231:
       vmul_231(a,b,p,N);
       vsum_231(a,b,s,N);
       break;
     case 232:
       vmul_232(a,b,p,N);
       vsum_232(a,b,s,N);
       break;
     case 233:
       vmul_233(a,b,p,N);
       vsum_233(a,b,s,N);
       break;
     case 234:
       vmul_234(a,b,p,N);
       vsum_234(a,b,s,N);
       break;
     case 235:
       vmul_235(a,b,p,N);
       vsum_235(a,b,s,N);
       break;
     case 236:
       vmul_236(a,b,p,N);
       vsum_236(a,b,s,N);
       break;
     case 237:
       vmul_237(a,b,p,N);
       vsum_237(a,b,s,N);
       break;
     case 238:
       vmul_238(a,b,p,N);
       vsum_238(a,b,s,N);
       break;
     case 239:
       vmul_239(a,b,p,N);
       vsum_239(a,b,s,N);
       break;
     case 240:
       vmul_240(a,b,p,N);
       vsum_240(a,b,s,N);
       break;
     case 241:
       vmul_241(a,b,p,N);
       vsum_241(a,b,s,N);
       break;
     case 242:
       vmul_242(a,b,p,N);
       vsum_242(a,b,s,N);
       break;
     case 243:
       vmul_243(a,b,p,N);
       vsum_243(a,b,s,N);
       break;
     case 244:
       vmul_244(a,b,p,N);
       vsum_244(a,b,s,N);
       break;
     case 245:
       vmul_245(a,b,p,N);
       vsum_245(a,b,s,N);
       break;
     case 246:
       vmul_246(a,b,p,N);
       vsum_246(a,b,s,N);
       break;
     case 247:
       vmul_247(a,b,p,N);
       vsum_247(a,b,s,N);
       break;
     case 248:
       vmul_248(a,b,p,N);
       vsum_248(a,b,s,N);
       break;
     case 249:
       vmul_249(a,b,p,N);
       vsum_249(a,b,s,N);
       break;
     case 250:
       vmul_250(a,b,p,N);
       vsum_250(a,b,s,N);
       break;
     case 251:
       vmul_251(a,b,p,N);
       vsum_251(a,b,s,N);
       break;
     case 252:
       vmul_252(a,b,p,N);
       vsum_252(a,b,s,N);
       break;
     case 253:
       vmul_253(a,b,p,N);
       vsum_253(a,b,s,N);
       break;
     case 254:
       vmul_254(a,b,p,N);
       vsum_254(a,b,s,N);
       break;
     case 255:
       vmul_255(a,b,p,N);
       vsum_255(a,b,s,N);
       break;
     case 256:
       vmul_256(a,b,p,N);
       vsum_256(a,b,s,N);
       break;
     case 257:
       vmul_257(a,b,p,N);
       vsum_257(a,b,s,N);
       break;
     case 258:
       vmul_258(a,b,p,N);
       vsum_258(a,b,s,N);
       break;
     case 259:
       vmul_259(a,b,p,N);
       vsum_259(a,b,s,N);
       break;
     case 260:
       vmul_260(a,b,p,N);
       vsum_260(a,b,s,N);
       break;
     case 261:
       vmul_261(a,b,p,N);
       vsum_261(a,b,s,N);
       break;
     case 262:
       vmul_262(a,b,p,N);
       vsum_262(a,b,s,N);
       break;
     case 263:
       vmul_263(a,b,p,N);
       vsum_263(a,b,s,N);
       break;
     case 264:
       vmul_264(a,b,p,N);
       vsum_264(a,b,s,N);
       break;
     case 265:
       vmul_265(a,b,p,N);
       vsum_265(a,b,s,N);
       break;
     case 266:
       vmul_266(a,b,p,N);
       vsum_266(a,b,s,N);
       break;
     case 267:
       vmul_267(a,b,p,N);
       vsum_267(a,b,s,N);
       break;
     case 268:
       vmul_268(a,b,p,N);
       vsum_268(a,b,s,N);
       break;
     case 269:
       vmul_269(a,b,p,N);
       vsum_269(a,b,s,N);
       break;
     case 270:
       vmul_270(a,b,p,N);
       vsum_270(a,b,s,N);
       break;
     case 271:
       vmul_271(a,b,p,N);
       vsum_271(a,b,s,N);
       break;
     case 272:
       vmul_272(a,b,p,N);
       vsum_272(a,b,s,N);
       break;
     case 273:
       vmul_273(a,b,p,N);
       vsum_273(a,b,s,N);
       break;
     case 274:
       vmul_274(a,b,p,N);
       vsum_274(a,b,s,N);
       break;
     case 275:
       vmul_275(a,b,p,N);
       vsum_275(a,b,s,N);
       break;
     case 276:
       vmul_276(a,b,p,N);
       vsum_276(a,b,s,N);
       break;
     case 277:
       vmul_277(a,b,p,N);
       vsum_277(a,b,s,N);
       break;
     case 278:
       vmul_278(a,b,p,N);
       vsum_278(a,b,s,N);
       break;
     case 279:
       vmul_279(a,b,p,N);
       vsum_279(a,b,s,N);
       break;
     case 280:
       vmul_280(a,b,p,N);
       vsum_280(a,b,s,N);
       break;
     case 281:
       vmul_281(a,b,p,N);
       vsum_281(a,b,s,N);
       break;
     case 282:
       vmul_282(a,b,p,N);
       vsum_282(a,b,s,N);
       break;
     case 283:
       vmul_283(a,b,p,N);
       vsum_283(a,b,s,N);
       break;
     case 284:
       vmul_284(a,b,p,N);
       vsum_284(a,b,s,N);
       break;
     case 285:
       vmul_285(a,b,p,N);
       vsum_285(a,b,s,N);
       break;
     case 286:
       vmul_286(a,b,p,N);
       vsum_286(a,b,s,N);
       break;
     case 287:
       vmul_287(a,b,p,N);
       vsum_287(a,b,s,N);
       break;
     case 288:
       vmul_288(a,b,p,N);
       vsum_288(a,b,s,N);
       break;
     case 289:
       vmul_289(a,b,p,N);
       vsum_289(a,b,s,N);
       break;
     case 290:
       vmul_290(a,b,p,N);
       vsum_290(a,b,s,N);
       break;
     case 291:
       vmul_291(a,b,p,N);
       vsum_291(a,b,s,N);
       break;
     case 292:
       vmul_292(a,b,p,N);
       vsum_292(a,b,s,N);
       break;
     case 293:
       vmul_293(a,b,p,N);
       vsum_293(a,b,s,N);
       break;
     case 294:
       vmul_294(a,b,p,N);
       vsum_294(a,b,s,N);
       break;
     case 295:
       vmul_295(a,b,p,N);
       vsum_295(a,b,s,N);
       break;
     case 296:
       vmul_296(a,b,p,N);
       vsum_296(a,b,s,N);
       break;
     case 297:
       vmul_297(a,b,p,N);
       vsum_297(a,b,s,N);
       break;
     case 298:
       vmul_298(a,b,p,N);
       vsum_298(a,b,s,N);
       break;
     case 299:
       vmul_299(a,b,p,N);
       vsum_299(a,b,s,N);
       break;
     case 300:
       vmul_300(a,b,p,N);
       vsum_300(a,b,s,N);
       break;
     case 301:
       vmul_301(a,b,p,N);
       vsum_301(a,b,s,N);
       break;
     case 302:
       vmul_302(a,b,p,N);
       vsum_302(a,b,s,N);
       break;
     case 303:
       vmul_303(a,b,p,N);
       vsum_303(a,b,s,N);
       break;
     case 304:
       vmul_304(a,b,p,N);
       vsum_304(a,b,s,N);
       break;
     case 305:
       vmul_305(a,b,p,N);
       vsum_305(a,b,s,N);
       break;
     case 306:
       vmul_306(a,b,p,N);
       vsum_306(a,b,s,N);
       break;
     case 307:
       vmul_307(a,b,p,N);
       vsum_307(a,b,s,N);
       break;
     case 308:
       vmul_308(a,b,p,N);
       vsum_308(a,b,s,N);
       break;
     case 309:
       vmul_309(a,b,p,N);
       vsum_309(a,b,s,N);
       break;
     case 310:
       vmul_310(a,b,p,N);
       vsum_310(a,b,s,N);
       break;
     case 311:
       vmul_311(a,b,p,N);
       vsum_311(a,b,s,N);
       break;
     case 312:
       vmul_312(a,b,p,N);
       vsum_312(a,b,s,N);
       break;
     case 313:
       vmul_313(a,b,p,N);
       vsum_313(a,b,s,N);
       break;
     case 314:
       vmul_314(a,b,p,N);
       vsum_314(a,b,s,N);
       break;
     case 315:
       vmul_315(a,b,p,N);
       vsum_315(a,b,s,N);
       break;
     case 316:
       vmul_316(a,b,p,N);
       vsum_316(a,b,s,N);
       break;
     case 317:
       vmul_317(a,b,p,N);
       vsum_317(a,b,s,N);
       break;
     case 318:
       vmul_318(a,b,p,N);
       vsum_318(a,b,s,N);
       break;
     case 319:
       vmul_319(a,b,p,N);
       vsum_319(a,b,s,N);
       break;
     case 320:
       vmul_320(a,b,p,N);
       vsum_320(a,b,s,N);
       break;
     case 321:
       vmul_321(a,b,p,N);
       vsum_321(a,b,s,N);
       break;
     case 322:
       vmul_322(a,b,p,N);
       vsum_322(a,b,s,N);
       break;
     case 323:
       vmul_323(a,b,p,N);
       vsum_323(a,b,s,N);
       break;
     case 324:
       vmul_324(a,b,p,N);
       vsum_324(a,b,s,N);
       break;
     case 325:
       vmul_325(a,b,p,N);
       vsum_325(a,b,s,N);
       break;
     case 326:
       vmul_326(a,b,p,N);
       vsum_326(a,b,s,N);
       break;
     case 327:
       vmul_327(a,b,p,N);
       vsum_327(a,b,s,N);
       break;
     case 328:
       vmul_328(a,b,p,N);
       vsum_328(a,b,s,N);
       break;
     case 329:
       vmul_329(a,b,p,N);
       vsum_329(a,b,s,N);
       break;
     case 330:
       vmul_330(a,b,p,N);
       vsum_330(a,b,s,N);
       break;
     case 331:
       vmul_331(a,b,p,N);
       vsum_331(a,b,s,N);
       break;
     case 332:
       vmul_332(a,b,p,N);
       vsum_332(a,b,s,N);
       break;
     case 333:
       vmul_333(a,b,p,N);
       vsum_333(a,b,s,N);
       break;
     case 334:
       vmul_334(a,b,p,N);
       vsum_334(a,b,s,N);
       break;
     case 335:
       vmul_335(a,b,p,N);
       vsum_335(a,b,s,N);
       break;
     case 336:
       vmul_336(a,b,p,N);
       vsum_336(a,b,s,N);
       break;
     case 337:
       vmul_337(a,b,p,N);
       vsum_337(a,b,s,N);
       break;
     case 338:
       vmul_338(a,b,p,N);
       vsum_338(a,b,s,N);
       break;
     case 339:
       vmul_339(a,b,p,N);
       vsum_339(a,b,s,N);
       break;
     case 340:
       vmul_340(a,b,p,N);
       vsum_340(a,b,s,N);
       break;
     case 341:
       vmul_341(a,b,p,N);
       vsum_341(a,b,s,N);
       break;
     case 342:
       vmul_342(a,b,p,N);
       vsum_342(a,b,s,N);
       break;
     case 343:
       vmul_343(a,b,p,N);
       vsum_343(a,b,s,N);
       break;
     case 344:
       vmul_344(a,b,p,N);
       vsum_344(a,b,s,N);
       break;
     case 345:
       vmul_345(a,b,p,N);
       vsum_345(a,b,s,N);
       break;
     case 346:
       vmul_346(a,b,p,N);
       vsum_346(a,b,s,N);
       break;
     case 347:
       vmul_347(a,b,p,N);
       vsum_347(a,b,s,N);
       break;
     case 348:
       vmul_348(a,b,p,N);
       vsum_348(a,b,s,N);
       break;
     case 349:
       vmul_349(a,b,p,N);
       vsum_349(a,b,s,N);
       break;
     case 350:
       vmul_350(a,b,p,N);
       vsum_350(a,b,s,N);
       break;
     case 351:
       vmul_351(a,b,p,N);
       vsum_351(a,b,s,N);
       break;
     case 352:
       vmul_352(a,b,p,N);
       vsum_352(a,b,s,N);
       break;
     case 353:
       vmul_353(a,b,p,N);
       vsum_353(a,b,s,N);
       break;
     case 354:
       vmul_354(a,b,p,N);
       vsum_354(a,b,s,N);
       break;
     case 355:
       vmul_355(a,b,p,N);
       vsum_355(a,b,s,N);
       break;
     case 356:
       vmul_356(a,b,p,N);
       vsum_356(a,b,s,N);
       break;
     case 357:
       vmul_357(a,b,p,N);
       vsum_357(a,b,s,N);
       break;
     case 358:
       vmul_358(a,b,p,N);
       vsum_358(a,b,s,N);
       break;
     case 359:
       vmul_359(a,b,p,N);
       vsum_359(a,b,s,N);
       break;
     case 360:
       vmul_360(a,b,p,N);
       vsum_360(a,b,s,N);
       break;
     case 361:
       vmul_361(a,b,p,N);
       vsum_361(a,b,s,N);
       break;
     case 362:
       vmul_362(a,b,p,N);
       vsum_362(a,b,s,N);
       break;
     case 363:
       vmul_363(a,b,p,N);
       vsum_363(a,b,s,N);
       break;
     case 364:
       vmul_364(a,b,p,N);
       vsum_364(a,b,s,N);
       break;
     case 365:
       vmul_365(a,b,p,N);
       vsum_365(a,b,s,N);
       break;
     case 366:
       vmul_366(a,b,p,N);
       vsum_366(a,b,s,N);
       break;
     case 367:
       vmul_367(a,b,p,N);
       vsum_367(a,b,s,N);
       break;
     case 368:
       vmul_368(a,b,p,N);
       vsum_368(a,b,s,N);
       break;
     case 369:
       vmul_369(a,b,p,N);
       vsum_369(a,b,s,N);
       break;
     case 370:
       vmul_370(a,b,p,N);
       vsum_370(a,b,s,N);
       break;
     case 371:
       vmul_371(a,b,p,N);
       vsum_371(a,b,s,N);
       break;
     case 372:
       vmul_372(a,b,p,N);
       vsum_372(a,b,s,N);
       break;
     case 373:
       vmul_373(a,b,p,N);
       vsum_373(a,b,s,N);
       break;
     case 374:
       vmul_374(a,b,p,N);
       vsum_374(a,b,s,N);
       break;
     case 375:
       vmul_375(a,b,p,N);
       vsum_375(a,b,s,N);
       break;
     case 376:
       vmul_376(a,b,p,N);
       vsum_376(a,b,s,N);
       break;
     case 377:
       vmul_377(a,b,p,N);
       vsum_377(a,b,s,N);
       break;
     case 378:
       vmul_378(a,b,p,N);
       vsum_378(a,b,s,N);
       break;
     case 379:
       vmul_379(a,b,p,N);
       vsum_379(a,b,s,N);
       break;
     case 380:
       vmul_380(a,b,p,N);
       vsum_380(a,b,s,N);
       break;
     case 381:
       vmul_381(a,b,p,N);
       vsum_381(a,b,s,N);
       break;
     case 382:
       vmul_382(a,b,p,N);
       vsum_382(a,b,s,N);
       break;
     case 383:
       vmul_383(a,b,p,N);
       vsum_383(a,b,s,N);
       break;
     case 384:
       vmul_384(a,b,p,N);
       vsum_384(a,b,s,N);
       break;
     case 385:
       vmul_385(a,b,p,N);
       vsum_385(a,b,s,N);
       break;
     case 386:
       vmul_386(a,b,p,N);
       vsum_386(a,b,s,N);
       break;
     case 387:
       vmul_387(a,b,p,N);
       vsum_387(a,b,s,N);
       break;
     case 388:
       vmul_388(a,b,p,N);
       vsum_388(a,b,s,N);
       break;
     case 389:
       vmul_389(a,b,p,N);
       vsum_389(a,b,s,N);
       break;
     case 390:
       vmul_390(a,b,p,N);
       vsum_390(a,b,s,N);
       break;
     case 391:
       vmul_391(a,b,p,N);
       vsum_391(a,b,s,N);
       break;
     case 392:
       vmul_392(a,b,p,N);
       vsum_392(a,b,s,N);
       break;
     case 393:
       vmul_393(a,b,p,N);
       vsum_393(a,b,s,N);
       break;
     case 394:
       vmul_394(a,b,p,N);
       vsum_394(a,b,s,N);
       break;
     case 395:
       vmul_395(a,b,p,N);
       vsum_395(a,b,s,N);
       break;
     case 396:
       vmul_396(a,b,p,N);
       vsum_396(a,b,s,N);
       break;
     case 397:
       vmul_397(a,b,p,N);
       vsum_397(a,b,s,N);
       break;
     case 398:
       vmul_398(a,b,p,N);
       vsum_398(a,b,s,N);
       break;
     case 399:
       vmul_399(a,b,p,N);
       vsum_399(a,b,s,N);
       break;
     case 400:
       vmul_400(a,b,p,N);
       vsum_400(a,b,s,N);
       break;
     case 401:
       vmul_401(a,b,p,N);
       vsum_401(a,b,s,N);
       break;
     case 402:
       vmul_402(a,b,p,N);
       vsum_402(a,b,s,N);
       break;
     case 403:
       vmul_403(a,b,p,N);
       vsum_403(a,b,s,N);
       break;
     case 404:
       vmul_404(a,b,p,N);
       vsum_404(a,b,s,N);
       break;
     case 405:
       vmul_405(a,b,p,N);
       vsum_405(a,b,s,N);
       break;
     case 406:
       vmul_406(a,b,p,N);
       vsum_406(a,b,s,N);
       break;
     case 407:
       vmul_407(a,b,p,N);
       vsum_407(a,b,s,N);
       break;
     case 408:
       vmul_408(a,b,p,N);
       vsum_408(a,b,s,N);
       break;
     case 409:
       vmul_409(a,b,p,N);
       vsum_409(a,b,s,N);
       break;
     case 410:
       vmul_410(a,b,p,N);
       vsum_410(a,b,s,N);
       break;
     case 411:
       vmul_411(a,b,p,N);
       vsum_411(a,b,s,N);
       break;
     case 412:
       vmul_412(a,b,p,N);
       vsum_412(a,b,s,N);
       break;
     case 413:
       vmul_413(a,b,p,N);
       vsum_413(a,b,s,N);
       break;
     case 414:
       vmul_414(a,b,p,N);
       vsum_414(a,b,s,N);
       break;
     case 415:
       vmul_415(a,b,p,N);
       vsum_415(a,b,s,N);
       break;
     case 416:
       vmul_416(a,b,p,N);
       vsum_416(a,b,s,N);
       break;
     case 417:
       vmul_417(a,b,p,N);
       vsum_417(a,b,s,N);
       break;
     case 418:
       vmul_418(a,b,p,N);
       vsum_418(a,b,s,N);
       break;
     case 419:
       vmul_419(a,b,p,N);
       vsum_419(a,b,s,N);
       break;
     case 420:
       vmul_420(a,b,p,N);
       vsum_420(a,b,s,N);
       break;
     case 421:
       vmul_421(a,b,p,N);
       vsum_421(a,b,s,N);
       break;
     case 422:
       vmul_422(a,b,p,N);
       vsum_422(a,b,s,N);
       break;
     case 423:
       vmul_423(a,b,p,N);
       vsum_423(a,b,s,N);
       break;
     case 424:
       vmul_424(a,b,p,N);
       vsum_424(a,b,s,N);
       break;
     case 425:
       vmul_425(a,b,p,N);
       vsum_425(a,b,s,N);
       break;
     case 426:
       vmul_426(a,b,p,N);
       vsum_426(a,b,s,N);
       break;
     case 427:
       vmul_427(a,b,p,N);
       vsum_427(a,b,s,N);
       break;
     case 428:
       vmul_428(a,b,p,N);
       vsum_428(a,b,s,N);
       break;
     case 429:
       vmul_429(a,b,p,N);
       vsum_429(a,b,s,N);
       break;
     case 430:
       vmul_430(a,b,p,N);
       vsum_430(a,b,s,N);
       break;
     case 431:
       vmul_431(a,b,p,N);
       vsum_431(a,b,s,N);
       break;
     case 432:
       vmul_432(a,b,p,N);
       vsum_432(a,b,s,N);
       break;
     case 433:
       vmul_433(a,b,p,N);
       vsum_433(a,b,s,N);
       break;
     case 434:
       vmul_434(a,b,p,N);
       vsum_434(a,b,s,N);
       break;
     case 435:
       vmul_435(a,b,p,N);
       vsum_435(a,b,s,N);
       break;
     case 436:
       vmul_436(a,b,p,N);
       vsum_436(a,b,s,N);
       break;
     case 437:
       vmul_437(a,b,p,N);
       vsum_437(a,b,s,N);
       break;
     case 438:
       vmul_438(a,b,p,N);
       vsum_438(a,b,s,N);
       break;
     case 439:
       vmul_439(a,b,p,N);
       vsum_439(a,b,s,N);
       break;
     case 440:
       vmul_440(a,b,p,N);
       vsum_440(a,b,s,N);
       break;
     case 441:
       vmul_441(a,b,p,N);
       vsum_441(a,b,s,N);
       break;
     case 442:
       vmul_442(a,b,p,N);
       vsum_442(a,b,s,N);
       break;
     case 443:
       vmul_443(a,b,p,N);
       vsum_443(a,b,s,N);
       break;
     case 444:
       vmul_444(a,b,p,N);
       vsum_444(a,b,s,N);
       break;
     case 445:
       vmul_445(a,b,p,N);
       vsum_445(a,b,s,N);
       break;
     case 446:
       vmul_446(a,b,p,N);
       vsum_446(a,b,s,N);
       break;
     case 447:
       vmul_447(a,b,p,N);
       vsum_447(a,b,s,N);
       break;
     case 448:
       vmul_448(a,b,p,N);
       vsum_448(a,b,s,N);
       break;
     case 449:
       vmul_449(a,b,p,N);
       vsum_449(a,b,s,N);
       break;
     case 450:
       vmul_450(a,b,p,N);
       vsum_450(a,b,s,N);
       break;
     case 451:
       vmul_451(a,b,p,N);
       vsum_451(a,b,s,N);
       break;
     case 452:
       vmul_452(a,b,p,N);
       vsum_452(a,b,s,N);
       break;
     case 453:
       vmul_453(a,b,p,N);
       vsum_453(a,b,s,N);
       break;
     case 454:
       vmul_454(a,b,p,N);
       vsum_454(a,b,s,N);
       break;
     case 455:
       vmul_455(a,b,p,N);
       vsum_455(a,b,s,N);
       break;
     case 456:
       vmul_456(a,b,p,N);
       vsum_456(a,b,s,N);
       break;
     case 457:
       vmul_457(a,b,p,N);
       vsum_457(a,b,s,N);
       break;
     case 458:
       vmul_458(a,b,p,N);
       vsum_458(a,b,s,N);
       break;
     case 459:
       vmul_459(a,b,p,N);
       vsum_459(a,b,s,N);
       break;
     case 460:
       vmul_460(a,b,p,N);
       vsum_460(a,b,s,N);
       break;
     case 461:
       vmul_461(a,b,p,N);
       vsum_461(a,b,s,N);
       break;
     case 462:
       vmul_462(a,b,p,N);
       vsum_462(a,b,s,N);
       break;
     case 463:
       vmul_463(a,b,p,N);
       vsum_463(a,b,s,N);
       break;
     case 464:
       vmul_464(a,b,p,N);
       vsum_464(a,b,s,N);
       break;
     case 465:
       vmul_465(a,b,p,N);
       vsum_465(a,b,s,N);
       break;
     case 466:
       vmul_466(a,b,p,N);
       vsum_466(a,b,s,N);
       break;
     case 467:
       vmul_467(a,b,p,N);
       vsum_467(a,b,s,N);
       break;
     case 468:
       vmul_468(a,b,p,N);
       vsum_468(a,b,s,N);
       break;
     case 469:
       vmul_469(a,b,p,N);
       vsum_469(a,b,s,N);
       break;
     case 470:
       vmul_470(a,b,p,N);
       vsum_470(a,b,s,N);
       break;
     case 471:
       vmul_471(a,b,p,N);
       vsum_471(a,b,s,N);
       break;
     case 472:
       vmul_472(a,b,p,N);
       vsum_472(a,b,s,N);
       break;
     case 473:
       vmul_473(a,b,p,N);
       vsum_473(a,b,s,N);
       break;
     case 474:
       vmul_474(a,b,p,N);
       vsum_474(a,b,s,N);
       break;
     case 475:
       vmul_475(a,b,p,N);
       vsum_475(a,b,s,N);
       break;
     case 476:
       vmul_476(a,b,p,N);
       vsum_476(a,b,s,N);
       break;
     case 477:
       vmul_477(a,b,p,N);
       vsum_477(a,b,s,N);
       break;
     case 478:
       vmul_478(a,b,p,N);
       vsum_478(a,b,s,N);
       break;
     case 479:
       vmul_479(a,b,p,N);
       vsum_479(a,b,s,N);
       break;
     case 480:
       vmul_480(a,b,p,N);
       vsum_480(a,b,s,N);
       break;
     case 481:
       vmul_481(a,b,p,N);
       vsum_481(a,b,s,N);
       break;
     case 482:
       vmul_482(a,b,p,N);
       vsum_482(a,b,s,N);
       break;
     case 483:
       vmul_483(a,b,p,N);
       vsum_483(a,b,s,N);
       break;
     case 484:
       vmul_484(a,b,p,N);
       vsum_484(a,b,s,N);
       break;
     case 485:
       vmul_485(a,b,p,N);
       vsum_485(a,b,s,N);
       break;
     case 486:
       vmul_486(a,b,p,N);
       vsum_486(a,b,s,N);
       break;
     case 487:
       vmul_487(a,b,p,N);
       vsum_487(a,b,s,N);
       break;
     case 488:
       vmul_488(a,b,p,N);
       vsum_488(a,b,s,N);
       break;
     case 489:
       vmul_489(a,b,p,N);
       vsum_489(a,b,s,N);
       break;
     case 490:
       vmul_490(a,b,p,N);
       vsum_490(a,b,s,N);
       break;
     case 491:
       vmul_491(a,b,p,N);
       vsum_491(a,b,s,N);
       break;
     case 492:
       vmul_492(a,b,p,N);
       vsum_492(a,b,s,N);
       break;
     case 493:
       vmul_493(a,b,p,N);
       vsum_493(a,b,s,N);
       break;
     case 494:
       vmul_494(a,b,p,N);
       vsum_494(a,b,s,N);
       break;
     case 495:
       vmul_495(a,b,p,N);
       vsum_495(a,b,s,N);
       break;
     case 496:
       vmul_496(a,b,p,N);
       vsum_496(a,b,s,N);
       break;
     case 497:
       vmul_497(a,b,p,N);
       vsum_497(a,b,s,N);
       break;
     case 498:
       vmul_498(a,b,p,N);
       vsum_498(a,b,s,N);
       break;
     case 499:
       vmul_499(a,b,p,N);
       vsum_499(a,b,s,N);
       break;
     case 500:
       vmul_500(a,b,p,N);
       vsum_500(a,b,s,N);
       break;
     case 501:
       vmul_501(a,b,p,N);
       vsum_501(a,b,s,N);
       break;
     case 502:
       vmul_502(a,b,p,N);
       vsum_502(a,b,s,N);
       break;
     case 503:
       vmul_503(a,b,p,N);
       vsum_503(a,b,s,N);
       break;
     case 504:
       vmul_504(a,b,p,N);
       vsum_504(a,b,s,N);
       break;
     case 505:
       vmul_505(a,b,p,N);
       vsum_505(a,b,s,N);
       break;
     case 506:
       vmul_506(a,b,p,N);
       vsum_506(a,b,s,N);
       break;
     case 507:
       vmul_507(a,b,p,N);
       vsum_507(a,b,s,N);
       break;
     case 508:
       vmul_508(a,b,p,N);
       vsum_508(a,b,s,N);
       break;
     case 509:
       vmul_509(a,b,p,N);
       vsum_509(a,b,s,N);
       break;
     case 510:
       vmul_510(a,b,p,N);
       vsum_510(a,b,s,N);
       break;
     case 511:
       vmul_511(a,b,p,N);
       vsum_511(a,b,s,N);
       break;
     case 512:
       vmul_512(a,b,p,N);
       vsum_512(a,b,s,N);
       break;
     case 513:
       vmul_513(a,b,p,N);
       vsum_513(a,b,s,N);
       break;
     case 514:
       vmul_514(a,b,p,N);
       vsum_514(a,b,s,N);
       break;
     case 515:
       vmul_515(a,b,p,N);
       vsum_515(a,b,s,N);
       break;
     case 516:
       vmul_516(a,b,p,N);
       vsum_516(a,b,s,N);
       break;
     case 517:
       vmul_517(a,b,p,N);
       vsum_517(a,b,s,N);
       break;
     case 518:
       vmul_518(a,b,p,N);
       vsum_518(a,b,s,N);
       break;
     case 519:
       vmul_519(a,b,p,N);
       vsum_519(a,b,s,N);
       break;
     case 520:
       vmul_520(a,b,p,N);
       vsum_520(a,b,s,N);
       break;
     case 521:
       vmul_521(a,b,p,N);
       vsum_521(a,b,s,N);
       break;
     case 522:
       vmul_522(a,b,p,N);
       vsum_522(a,b,s,N);
       break;
     case 523:
       vmul_523(a,b,p,N);
       vsum_523(a,b,s,N);
       break;
     case 524:
       vmul_524(a,b,p,N);
       vsum_524(a,b,s,N);
       break;
     case 525:
       vmul_525(a,b,p,N);
       vsum_525(a,b,s,N);
       break;
     case 526:
       vmul_526(a,b,p,N);
       vsum_526(a,b,s,N);
       break;
     case 527:
       vmul_527(a,b,p,N);
       vsum_527(a,b,s,N);
       break;
     case 528:
       vmul_528(a,b,p,N);
       vsum_528(a,b,s,N);
       break;
     case 529:
       vmul_529(a,b,p,N);
       vsum_529(a,b,s,N);
       break;
     case 530:
       vmul_530(a,b,p,N);
       vsum_530(a,b,s,N);
       break;
     case 531:
       vmul_531(a,b,p,N);
       vsum_531(a,b,s,N);
       break;
     case 532:
       vmul_532(a,b,p,N);
       vsum_532(a,b,s,N);
       break;
     case 533:
       vmul_533(a,b,p,N);
       vsum_533(a,b,s,N);
       break;
     case 534:
       vmul_534(a,b,p,N);
       vsum_534(a,b,s,N);
       break;
     case 535:
       vmul_535(a,b,p,N);
       vsum_535(a,b,s,N);
       break;
     case 536:
       vmul_536(a,b,p,N);
       vsum_536(a,b,s,N);
       break;
     case 537:
       vmul_537(a,b,p,N);
       vsum_537(a,b,s,N);
       break;
     case 538:
       vmul_538(a,b,p,N);
       vsum_538(a,b,s,N);
       break;
     case 539:
       vmul_539(a,b,p,N);
       vsum_539(a,b,s,N);
       break;
     case 540:
       vmul_540(a,b,p,N);
       vsum_540(a,b,s,N);
       break;
     case 541:
       vmul_541(a,b,p,N);
       vsum_541(a,b,s,N);
       break;
     case 542:
       vmul_542(a,b,p,N);
       vsum_542(a,b,s,N);
       break;
     case 543:
       vmul_543(a,b,p,N);
       vsum_543(a,b,s,N);
       break;
     case 544:
       vmul_544(a,b,p,N);
       vsum_544(a,b,s,N);
       break;
     case 545:
       vmul_545(a,b,p,N);
       vsum_545(a,b,s,N);
       break;
     case 546:
       vmul_546(a,b,p,N);
       vsum_546(a,b,s,N);
       break;
     case 547:
       vmul_547(a,b,p,N);
       vsum_547(a,b,s,N);
       break;
     case 548:
       vmul_548(a,b,p,N);
       vsum_548(a,b,s,N);
       break;
     case 549:
       vmul_549(a,b,p,N);
       vsum_549(a,b,s,N);
       break;
     case 550:
       vmul_550(a,b,p,N);
       vsum_550(a,b,s,N);
       break;
     case 551:
       vmul_551(a,b,p,N);
       vsum_551(a,b,s,N);
       break;
     case 552:
       vmul_552(a,b,p,N);
       vsum_552(a,b,s,N);
       break;
     case 553:
       vmul_553(a,b,p,N);
       vsum_553(a,b,s,N);
       break;
     case 554:
       vmul_554(a,b,p,N);
       vsum_554(a,b,s,N);
       break;
     case 555:
       vmul_555(a,b,p,N);
       vsum_555(a,b,s,N);
       break;
     case 556:
       vmul_556(a,b,p,N);
       vsum_556(a,b,s,N);
       break;
     case 557:
       vmul_557(a,b,p,N);
       vsum_557(a,b,s,N);
       break;
     case 558:
       vmul_558(a,b,p,N);
       vsum_558(a,b,s,N);
       break;
     case 559:
       vmul_559(a,b,p,N);
       vsum_559(a,b,s,N);
       break;
     case 560:
       vmul_560(a,b,p,N);
       vsum_560(a,b,s,N);
       break;
     case 561:
       vmul_561(a,b,p,N);
       vsum_561(a,b,s,N);
       break;
     case 562:
       vmul_562(a,b,p,N);
       vsum_562(a,b,s,N);
       break;
     case 563:
       vmul_563(a,b,p,N);
       vsum_563(a,b,s,N);
       break;
     case 564:
       vmul_564(a,b,p,N);
       vsum_564(a,b,s,N);
       break;
     case 565:
       vmul_565(a,b,p,N);
       vsum_565(a,b,s,N);
       break;
     case 566:
       vmul_566(a,b,p,N);
       vsum_566(a,b,s,N);
       break;
     case 567:
       vmul_567(a,b,p,N);
       vsum_567(a,b,s,N);
       break;
     case 568:
       vmul_568(a,b,p,N);
       vsum_568(a,b,s,N);
       break;
     case 569:
       vmul_569(a,b,p,N);
       vsum_569(a,b,s,N);
       break;
     case 570:
       vmul_570(a,b,p,N);
       vsum_570(a,b,s,N);
       break;
     case 571:
       vmul_571(a,b,p,N);
       vsum_571(a,b,s,N);
       break;
     case 572:
       vmul_572(a,b,p,N);
       vsum_572(a,b,s,N);
       break;
     case 573:
       vmul_573(a,b,p,N);
       vsum_573(a,b,s,N);
       break;
     case 574:
       vmul_574(a,b,p,N);
       vsum_574(a,b,s,N);
       break;
     case 575:
       vmul_575(a,b,p,N);
       vsum_575(a,b,s,N);
       break;
     case 576:
       vmul_576(a,b,p,N);
       vsum_576(a,b,s,N);
       break;
     case 577:
       vmul_577(a,b,p,N);
       vsum_577(a,b,s,N);
       break;
     case 578:
       vmul_578(a,b,p,N);
       vsum_578(a,b,s,N);
       break;
     case 579:
       vmul_579(a,b,p,N);
       vsum_579(a,b,s,N);
       break;
     case 580:
       vmul_580(a,b,p,N);
       vsum_580(a,b,s,N);
       break;
     case 581:
       vmul_581(a,b,p,N);
       vsum_581(a,b,s,N);
       break;
     case 582:
       vmul_582(a,b,p,N);
       vsum_582(a,b,s,N);
       break;
     case 583:
       vmul_583(a,b,p,N);
       vsum_583(a,b,s,N);
       break;
     case 584:
       vmul_584(a,b,p,N);
       vsum_584(a,b,s,N);
       break;
     case 585:
       vmul_585(a,b,p,N);
       vsum_585(a,b,s,N);
       break;
     case 586:
       vmul_586(a,b,p,N);
       vsum_586(a,b,s,N);
       break;
     case 587:
       vmul_587(a,b,p,N);
       vsum_587(a,b,s,N);
       break;
     case 588:
       vmul_588(a,b,p,N);
       vsum_588(a,b,s,N);
       break;
     case 589:
       vmul_589(a,b,p,N);
       vsum_589(a,b,s,N);
       break;
     case 590:
       vmul_590(a,b,p,N);
       vsum_590(a,b,s,N);
       break;
     case 591:
       vmul_591(a,b,p,N);
       vsum_591(a,b,s,N);
       break;
     case 592:
       vmul_592(a,b,p,N);
       vsum_592(a,b,s,N);
       break;
     case 593:
       vmul_593(a,b,p,N);
       vsum_593(a,b,s,N);
       break;
     case 594:
       vmul_594(a,b,p,N);
       vsum_594(a,b,s,N);
       break;
     case 595:
       vmul_595(a,b,p,N);
       vsum_595(a,b,s,N);
       break;
     case 596:
       vmul_596(a,b,p,N);
       vsum_596(a,b,s,N);
       break;
     case 597:
       vmul_597(a,b,p,N);
       vsum_597(a,b,s,N);
       break;
     case 598:
       vmul_598(a,b,p,N);
       vsum_598(a,b,s,N);
       break;
     case 599:
       vmul_599(a,b,p,N);
       vsum_599(a,b,s,N);
       break;
     case 600:
       vmul_600(a,b,p,N);
       vsum_600(a,b,s,N);
       break;
     case 601:
       vmul_601(a,b,p,N);
       vsum_601(a,b,s,N);
       break;
     case 602:
       vmul_602(a,b,p,N);
       vsum_602(a,b,s,N);
       break;
     case 603:
       vmul_603(a,b,p,N);
       vsum_603(a,b,s,N);
       break;
     case 604:
       vmul_604(a,b,p,N);
       vsum_604(a,b,s,N);
       break;
     case 605:
       vmul_605(a,b,p,N);
       vsum_605(a,b,s,N);
       break;
     case 606:
       vmul_606(a,b,p,N);
       vsum_606(a,b,s,N);
       break;
     case 607:
       vmul_607(a,b,p,N);
       vsum_607(a,b,s,N);
       break;
     case 608:
       vmul_608(a,b,p,N);
       vsum_608(a,b,s,N);
       break;
     case 609:
       vmul_609(a,b,p,N);
       vsum_609(a,b,s,N);
       break;
     case 610:
       vmul_610(a,b,p,N);
       vsum_610(a,b,s,N);
       break;
     case 611:
       vmul_611(a,b,p,N);
       vsum_611(a,b,s,N);
       break;
     case 612:
       vmul_612(a,b,p,N);
       vsum_612(a,b,s,N);
       break;
     case 613:
       vmul_613(a,b,p,N);
       vsum_613(a,b,s,N);
       break;
     case 614:
       vmul_614(a,b,p,N);
       vsum_614(a,b,s,N);
       break;
     case 615:
       vmul_615(a,b,p,N);
       vsum_615(a,b,s,N);
       break;
     case 616:
       vmul_616(a,b,p,N);
       vsum_616(a,b,s,N);
       break;
     case 617:
       vmul_617(a,b,p,N);
       vsum_617(a,b,s,N);
       break;
     case 618:
       vmul_618(a,b,p,N);
       vsum_618(a,b,s,N);
       break;
     case 619:
       vmul_619(a,b,p,N);
       vsum_619(a,b,s,N);
       break;
     case 620:
       vmul_620(a,b,p,N);
       vsum_620(a,b,s,N);
       break;
     case 621:
       vmul_621(a,b,p,N);
       vsum_621(a,b,s,N);
       break;
     case 622:
       vmul_622(a,b,p,N);
       vsum_622(a,b,s,N);
       break;
     case 623:
       vmul_623(a,b,p,N);
       vsum_623(a,b,s,N);
       break;
     case 624:
       vmul_624(a,b,p,N);
       vsum_624(a,b,s,N);
       break;
     case 625:
       vmul_625(a,b,p,N);
       vsum_625(a,b,s,N);
       break;
     case 626:
       vmul_626(a,b,p,N);
       vsum_626(a,b,s,N);
       break;
     case 627:
       vmul_627(a,b,p,N);
       vsum_627(a,b,s,N);
       break;
     case 628:
       vmul_628(a,b,p,N);
       vsum_628(a,b,s,N);
       break;
     case 629:
       vmul_629(a,b,p,N);
       vsum_629(a,b,s,N);
       break;
     case 630:
       vmul_630(a,b,p,N);
       vsum_630(a,b,s,N);
       break;
     case 631:
       vmul_631(a,b,p,N);
       vsum_631(a,b,s,N);
       break;
     case 632:
       vmul_632(a,b,p,N);
       vsum_632(a,b,s,N);
       break;
     case 633:
       vmul_633(a,b,p,N);
       vsum_633(a,b,s,N);
       break;
     case 634:
       vmul_634(a,b,p,N);
       vsum_634(a,b,s,N);
       break;
     case 635:
       vmul_635(a,b,p,N);
       vsum_635(a,b,s,N);
       break;
     case 636:
       vmul_636(a,b,p,N);
       vsum_636(a,b,s,N);
       break;
     case 637:
       vmul_637(a,b,p,N);
       vsum_637(a,b,s,N);
       break;
     case 638:
       vmul_638(a,b,p,N);
       vsum_638(a,b,s,N);
       break;
     case 639:
       vmul_639(a,b,p,N);
       vsum_639(a,b,s,N);
       break;
     case 640:
       vmul_640(a,b,p,N);
       vsum_640(a,b,s,N);
       break;
     case 641:
       vmul_641(a,b,p,N);
       vsum_641(a,b,s,N);
       break;
     case 642:
       vmul_642(a,b,p,N);
       vsum_642(a,b,s,N);
       break;
     case 643:
       vmul_643(a,b,p,N);
       vsum_643(a,b,s,N);
       break;
     case 644:
       vmul_644(a,b,p,N);
       vsum_644(a,b,s,N);
       break;
     case 645:
       vmul_645(a,b,p,N);
       vsum_645(a,b,s,N);
       break;
     case 646:
       vmul_646(a,b,p,N);
       vsum_646(a,b,s,N);
       break;
     case 647:
       vmul_647(a,b,p,N);
       vsum_647(a,b,s,N);
       break;
     case 648:
       vmul_648(a,b,p,N);
       vsum_648(a,b,s,N);
       break;
     case 649:
       vmul_649(a,b,p,N);
       vsum_649(a,b,s,N);
       break;
     case 650:
       vmul_650(a,b,p,N);
       vsum_650(a,b,s,N);
       break;
     case 651:
       vmul_651(a,b,p,N);
       vsum_651(a,b,s,N);
       break;
     case 652:
       vmul_652(a,b,p,N);
       vsum_652(a,b,s,N);
       break;
     case 653:
       vmul_653(a,b,p,N);
       vsum_653(a,b,s,N);
       break;
     case 654:
       vmul_654(a,b,p,N);
       vsum_654(a,b,s,N);
       break;
     case 655:
       vmul_655(a,b,p,N);
       vsum_655(a,b,s,N);
       break;
     case 656:
       vmul_656(a,b,p,N);
       vsum_656(a,b,s,N);
       break;
     case 657:
       vmul_657(a,b,p,N);
       vsum_657(a,b,s,N);
       break;
     case 658:
       vmul_658(a,b,p,N);
       vsum_658(a,b,s,N);
       break;
     case 659:
       vmul_659(a,b,p,N);
       vsum_659(a,b,s,N);
       break;
     case 660:
       vmul_660(a,b,p,N);
       vsum_660(a,b,s,N);
       break;
     case 661:
       vmul_661(a,b,p,N);
       vsum_661(a,b,s,N);
       break;
     case 662:
       vmul_662(a,b,p,N);
       vsum_662(a,b,s,N);
       break;
     case 663:
       vmul_663(a,b,p,N);
       vsum_663(a,b,s,N);
       break;
     case 664:
       vmul_664(a,b,p,N);
       vsum_664(a,b,s,N);
       break;
     case 665:
       vmul_665(a,b,p,N);
       vsum_665(a,b,s,N);
       break;
     case 666:
       vmul_666(a,b,p,N);
       vsum_666(a,b,s,N);
       break;
     case 667:
       vmul_667(a,b,p,N);
       vsum_667(a,b,s,N);
       break;
     case 668:
       vmul_668(a,b,p,N);
       vsum_668(a,b,s,N);
       break;
     case 669:
       vmul_669(a,b,p,N);
       vsum_669(a,b,s,N);
       break;
     case 670:
       vmul_670(a,b,p,N);
       vsum_670(a,b,s,N);
       break;
     case 671:
       vmul_671(a,b,p,N);
       vsum_671(a,b,s,N);
       break;
     case 672:
       vmul_672(a,b,p,N);
       vsum_672(a,b,s,N);
       break;
     case 673:
       vmul_673(a,b,p,N);
       vsum_673(a,b,s,N);
       break;
     case 674:
       vmul_674(a,b,p,N);
       vsum_674(a,b,s,N);
       break;
     case 675:
       vmul_675(a,b,p,N);
       vsum_675(a,b,s,N);
       break;
     case 676:
       vmul_676(a,b,p,N);
       vsum_676(a,b,s,N);
       break;
     case 677:
       vmul_677(a,b,p,N);
       vsum_677(a,b,s,N);
       break;
     case 678:
       vmul_678(a,b,p,N);
       vsum_678(a,b,s,N);
       break;
     case 679:
       vmul_679(a,b,p,N);
       vsum_679(a,b,s,N);
       break;
     case 680:
       vmul_680(a,b,p,N);
       vsum_680(a,b,s,N);
       break;
     case 681:
       vmul_681(a,b,p,N);
       vsum_681(a,b,s,N);
       break;
     case 682:
       vmul_682(a,b,p,N);
       vsum_682(a,b,s,N);
       break;
     case 683:
       vmul_683(a,b,p,N);
       vsum_683(a,b,s,N);
       break;
     case 684:
       vmul_684(a,b,p,N);
       vsum_684(a,b,s,N);
       break;
     case 685:
       vmul_685(a,b,p,N);
       vsum_685(a,b,s,N);
       break;
     case 686:
       vmul_686(a,b,p,N);
       vsum_686(a,b,s,N);
       break;
     case 687:
       vmul_687(a,b,p,N);
       vsum_687(a,b,s,N);
       break;
     case 688:
       vmul_688(a,b,p,N);
       vsum_688(a,b,s,N);
       break;
     case 689:
       vmul_689(a,b,p,N);
       vsum_689(a,b,s,N);
       break;
     case 690:
       vmul_690(a,b,p,N);
       vsum_690(a,b,s,N);
       break;
     case 691:
       vmul_691(a,b,p,N);
       vsum_691(a,b,s,N);
       break;
     case 692:
       vmul_692(a,b,p,N);
       vsum_692(a,b,s,N);
       break;
     case 693:
       vmul_693(a,b,p,N);
       vsum_693(a,b,s,N);
       break;
     case 694:
       vmul_694(a,b,p,N);
       vsum_694(a,b,s,N);
       break;
     case 695:
       vmul_695(a,b,p,N);
       vsum_695(a,b,s,N);
       break;
     case 696:
       vmul_696(a,b,p,N);
       vsum_696(a,b,s,N);
       break;
     case 697:
       vmul_697(a,b,p,N);
       vsum_697(a,b,s,N);
       break;
     case 698:
       vmul_698(a,b,p,N);
       vsum_698(a,b,s,N);
       break;
     case 699:
       vmul_699(a,b,p,N);
       vsum_699(a,b,s,N);
       break;
     case 700:
       vmul_700(a,b,p,N);
       vsum_700(a,b,s,N);
       break;
     case 701:
       vmul_701(a,b,p,N);
       vsum_701(a,b,s,N);
       break;
     case 702:
       vmul_702(a,b,p,N);
       vsum_702(a,b,s,N);
       break;
     case 703:
       vmul_703(a,b,p,N);
       vsum_703(a,b,s,N);
       break;
     case 704:
       vmul_704(a,b,p,N);
       vsum_704(a,b,s,N);
       break;
     case 705:
       vmul_705(a,b,p,N);
       vsum_705(a,b,s,N);
       break;
     case 706:
       vmul_706(a,b,p,N);
       vsum_706(a,b,s,N);
       break;
     case 707:
       vmul_707(a,b,p,N);
       vsum_707(a,b,s,N);
       break;
     case 708:
       vmul_708(a,b,p,N);
       vsum_708(a,b,s,N);
       break;
     case 709:
       vmul_709(a,b,p,N);
       vsum_709(a,b,s,N);
       break;
     case 710:
       vmul_710(a,b,p,N);
       vsum_710(a,b,s,N);
       break;
     case 711:
       vmul_711(a,b,p,N);
       vsum_711(a,b,s,N);
       break;
     case 712:
       vmul_712(a,b,p,N);
       vsum_712(a,b,s,N);
       break;
     case 713:
       vmul_713(a,b,p,N);
       vsum_713(a,b,s,N);
       break;
     case 714:
       vmul_714(a,b,p,N);
       vsum_714(a,b,s,N);
       break;
     case 715:
       vmul_715(a,b,p,N);
       vsum_715(a,b,s,N);
       break;
     case 716:
       vmul_716(a,b,p,N);
       vsum_716(a,b,s,N);
       break;
     case 717:
       vmul_717(a,b,p,N);
       vsum_717(a,b,s,N);
       break;
     case 718:
       vmul_718(a,b,p,N);
       vsum_718(a,b,s,N);
       break;
     case 719:
       vmul_719(a,b,p,N);
       vsum_719(a,b,s,N);
       break;
     case 720:
       vmul_720(a,b,p,N);
       vsum_720(a,b,s,N);
       break;
     case 721:
       vmul_721(a,b,p,N);
       vsum_721(a,b,s,N);
       break;
     case 722:
       vmul_722(a,b,p,N);
       vsum_722(a,b,s,N);
       break;
     case 723:
       vmul_723(a,b,p,N);
       vsum_723(a,b,s,N);
       break;
     case 724:
       vmul_724(a,b,p,N);
       vsum_724(a,b,s,N);
       break;
     case 725:
       vmul_725(a,b,p,N);
       vsum_725(a,b,s,N);
       break;
     case 726:
       vmul_726(a,b,p,N);
       vsum_726(a,b,s,N);
       break;
     case 727:
       vmul_727(a,b,p,N);
       vsum_727(a,b,s,N);
       break;
     case 728:
       vmul_728(a,b,p,N);
       vsum_728(a,b,s,N);
       break;
     case 729:
       vmul_729(a,b,p,N);
       vsum_729(a,b,s,N);
       break;
     case 730:
       vmul_730(a,b,p,N);
       vsum_730(a,b,s,N);
       break;
     case 731:
       vmul_731(a,b,p,N);
       vsum_731(a,b,s,N);
       break;
     case 732:
       vmul_732(a,b,p,N);
       vsum_732(a,b,s,N);
       break;
     case 733:
       vmul_733(a,b,p,N);
       vsum_733(a,b,s,N);
       break;
     case 734:
       vmul_734(a,b,p,N);
       vsum_734(a,b,s,N);
       break;
     case 735:
       vmul_735(a,b,p,N);
       vsum_735(a,b,s,N);
       break;
     case 736:
       vmul_736(a,b,p,N);
       vsum_736(a,b,s,N);
       break;
     case 737:
       vmul_737(a,b,p,N);
       vsum_737(a,b,s,N);
       break;
     case 738:
       vmul_738(a,b,p,N);
       vsum_738(a,b,s,N);
       break;
     case 739:
       vmul_739(a,b,p,N);
       vsum_739(a,b,s,N);
       break;
     case 740:
       vmul_740(a,b,p,N);
       vsum_740(a,b,s,N);
       break;
     case 741:
       vmul_741(a,b,p,N);
       vsum_741(a,b,s,N);
       break;
     case 742:
       vmul_742(a,b,p,N);
       vsum_742(a,b,s,N);
       break;
     case 743:
       vmul_743(a,b,p,N);
       vsum_743(a,b,s,N);
       break;
     case 744:
       vmul_744(a,b,p,N);
       vsum_744(a,b,s,N);
       break;
     case 745:
       vmul_745(a,b,p,N);
       vsum_745(a,b,s,N);
       break;
     case 746:
       vmul_746(a,b,p,N);
       vsum_746(a,b,s,N);
       break;
     case 747:
       vmul_747(a,b,p,N);
       vsum_747(a,b,s,N);
       break;
     case 748:
       vmul_748(a,b,p,N);
       vsum_748(a,b,s,N);
       break;
     case 749:
       vmul_749(a,b,p,N);
       vsum_749(a,b,s,N);
       break;
     case 750:
       vmul_750(a,b,p,N);
       vsum_750(a,b,s,N);
       break;
     case 751:
       vmul_751(a,b,p,N);
       vsum_751(a,b,s,N);
       break;
     case 752:
       vmul_752(a,b,p,N);
       vsum_752(a,b,s,N);
       break;
     case 753:
       vmul_753(a,b,p,N);
       vsum_753(a,b,s,N);
       break;
     case 754:
       vmul_754(a,b,p,N);
       vsum_754(a,b,s,N);
       break;
     case 755:
       vmul_755(a,b,p,N);
       vsum_755(a,b,s,N);
       break;
     case 756:
       vmul_756(a,b,p,N);
       vsum_756(a,b,s,N);
       break;
     case 757:
       vmul_757(a,b,p,N);
       vsum_757(a,b,s,N);
       break;
     case 758:
       vmul_758(a,b,p,N);
       vsum_758(a,b,s,N);
       break;
     case 759:
       vmul_759(a,b,p,N);
       vsum_759(a,b,s,N);
       break;
     case 760:
       vmul_760(a,b,p,N);
       vsum_760(a,b,s,N);
       break;
     case 761:
       vmul_761(a,b,p,N);
       vsum_761(a,b,s,N);
       break;
     case 762:
       vmul_762(a,b,p,N);
       vsum_762(a,b,s,N);
       break;
     case 763:
       vmul_763(a,b,p,N);
       vsum_763(a,b,s,N);
       break;
     case 764:
       vmul_764(a,b,p,N);
       vsum_764(a,b,s,N);
       break;
     case 765:
       vmul_765(a,b,p,N);
       vsum_765(a,b,s,N);
       break;
     case 766:
       vmul_766(a,b,p,N);
       vsum_766(a,b,s,N);
       break;
     case 767:
       vmul_767(a,b,p,N);
       vsum_767(a,b,s,N);
       break;
     case 768:
       vmul_768(a,b,p,N);
       vsum_768(a,b,s,N);
       break;
     case 769:
       vmul_769(a,b,p,N);
       vsum_769(a,b,s,N);
       break;
     case 770:
       vmul_770(a,b,p,N);
       vsum_770(a,b,s,N);
       break;
     case 771:
       vmul_771(a,b,p,N);
       vsum_771(a,b,s,N);
       break;
     case 772:
       vmul_772(a,b,p,N);
       vsum_772(a,b,s,N);
       break;
     case 773:
       vmul_773(a,b,p,N);
       vsum_773(a,b,s,N);
       break;
     case 774:
       vmul_774(a,b,p,N);
       vsum_774(a,b,s,N);
       break;
     case 775:
       vmul_775(a,b,p,N);
       vsum_775(a,b,s,N);
       break;
     case 776:
       vmul_776(a,b,p,N);
       vsum_776(a,b,s,N);
       break;
     case 777:
       vmul_777(a,b,p,N);
       vsum_777(a,b,s,N);
       break;
     case 778:
       vmul_778(a,b,p,N);
       vsum_778(a,b,s,N);
       break;
     case 779:
       vmul_779(a,b,p,N);
       vsum_779(a,b,s,N);
       break;
     case 780:
       vmul_780(a,b,p,N);
       vsum_780(a,b,s,N);
       break;
     case 781:
       vmul_781(a,b,p,N);
       vsum_781(a,b,s,N);
       break;
     case 782:
       vmul_782(a,b,p,N);
       vsum_782(a,b,s,N);
       break;
     case 783:
       vmul_783(a,b,p,N);
       vsum_783(a,b,s,N);
       break;
     case 784:
       vmul_784(a,b,p,N);
       vsum_784(a,b,s,N);
       break;
     case 785:
       vmul_785(a,b,p,N);
       vsum_785(a,b,s,N);
       break;
     case 786:
       vmul_786(a,b,p,N);
       vsum_786(a,b,s,N);
       break;
     case 787:
       vmul_787(a,b,p,N);
       vsum_787(a,b,s,N);
       break;
     case 788:
       vmul_788(a,b,p,N);
       vsum_788(a,b,s,N);
       break;
     case 789:
       vmul_789(a,b,p,N);
       vsum_789(a,b,s,N);
       break;
     case 790:
       vmul_790(a,b,p,N);
       vsum_790(a,b,s,N);
       break;
     case 791:
       vmul_791(a,b,p,N);
       vsum_791(a,b,s,N);
       break;
     case 792:
       vmul_792(a,b,p,N);
       vsum_792(a,b,s,N);
       break;
     case 793:
       vmul_793(a,b,p,N);
       vsum_793(a,b,s,N);
       break;
     case 794:
       vmul_794(a,b,p,N);
       vsum_794(a,b,s,N);
       break;
     case 795:
       vmul_795(a,b,p,N);
       vsum_795(a,b,s,N);
       break;
     case 796:
       vmul_796(a,b,p,N);
       vsum_796(a,b,s,N);
       break;
     case 797:
       vmul_797(a,b,p,N);
       vsum_797(a,b,s,N);
       break;
     case 798:
       vmul_798(a,b,p,N);
       vsum_798(a,b,s,N);
       break;
     case 799:
       vmul_799(a,b,p,N);
       vsum_799(a,b,s,N);
       break;
     case 800:
       vmul_800(a,b,p,N);
       vsum_800(a,b,s,N);
       break;
     case 801:
       vmul_801(a,b,p,N);
       vsum_801(a,b,s,N);
       break;
     case 802:
       vmul_802(a,b,p,N);
       vsum_802(a,b,s,N);
       break;
     case 803:
       vmul_803(a,b,p,N);
       vsum_803(a,b,s,N);
       break;
     case 804:
       vmul_804(a,b,p,N);
       vsum_804(a,b,s,N);
       break;
     case 805:
       vmul_805(a,b,p,N);
       vsum_805(a,b,s,N);
       break;
     case 806:
       vmul_806(a,b,p,N);
       vsum_806(a,b,s,N);
       break;
     case 807:
       vmul_807(a,b,p,N);
       vsum_807(a,b,s,N);
       break;
     case 808:
       vmul_808(a,b,p,N);
       vsum_808(a,b,s,N);
       break;
     case 809:
       vmul_809(a,b,p,N);
       vsum_809(a,b,s,N);
       break;
     case 810:
       vmul_810(a,b,p,N);
       vsum_810(a,b,s,N);
       break;
     case 811:
       vmul_811(a,b,p,N);
       vsum_811(a,b,s,N);
       break;
     case 812:
       vmul_812(a,b,p,N);
       vsum_812(a,b,s,N);
       break;
     case 813:
       vmul_813(a,b,p,N);
       vsum_813(a,b,s,N);
       break;
     case 814:
       vmul_814(a,b,p,N);
       vsum_814(a,b,s,N);
       break;
     case 815:
       vmul_815(a,b,p,N);
       vsum_815(a,b,s,N);
       break;
     case 816:
       vmul_816(a,b,p,N);
       vsum_816(a,b,s,N);
       break;
     case 817:
       vmul_817(a,b,p,N);
       vsum_817(a,b,s,N);
       break;
     case 818:
       vmul_818(a,b,p,N);
       vsum_818(a,b,s,N);
       break;
     case 819:
       vmul_819(a,b,p,N);
       vsum_819(a,b,s,N);
       break;
     case 820:
       vmul_820(a,b,p,N);
       vsum_820(a,b,s,N);
       break;
     case 821:
       vmul_821(a,b,p,N);
       vsum_821(a,b,s,N);
       break;
     case 822:
       vmul_822(a,b,p,N);
       vsum_822(a,b,s,N);
       break;
     case 823:
       vmul_823(a,b,p,N);
       vsum_823(a,b,s,N);
       break;
     case 824:
       vmul_824(a,b,p,N);
       vsum_824(a,b,s,N);
       break;
     case 825:
       vmul_825(a,b,p,N);
       vsum_825(a,b,s,N);
       break;
     case 826:
       vmul_826(a,b,p,N);
       vsum_826(a,b,s,N);
       break;
     case 827:
       vmul_827(a,b,p,N);
       vsum_827(a,b,s,N);
       break;
     case 828:
       vmul_828(a,b,p,N);
       vsum_828(a,b,s,N);
       break;
     case 829:
       vmul_829(a,b,p,N);
       vsum_829(a,b,s,N);
       break;
     case 830:
       vmul_830(a,b,p,N);
       vsum_830(a,b,s,N);
       break;
     case 831:
       vmul_831(a,b,p,N);
       vsum_831(a,b,s,N);
       break;
     case 832:
       vmul_832(a,b,p,N);
       vsum_832(a,b,s,N);
       break;
     case 833:
       vmul_833(a,b,p,N);
       vsum_833(a,b,s,N);
       break;
     case 834:
       vmul_834(a,b,p,N);
       vsum_834(a,b,s,N);
       break;
     case 835:
       vmul_835(a,b,p,N);
       vsum_835(a,b,s,N);
       break;
     case 836:
       vmul_836(a,b,p,N);
       vsum_836(a,b,s,N);
       break;
     case 837:
       vmul_837(a,b,p,N);
       vsum_837(a,b,s,N);
       break;
     case 838:
       vmul_838(a,b,p,N);
       vsum_838(a,b,s,N);
       break;
     case 839:
       vmul_839(a,b,p,N);
       vsum_839(a,b,s,N);
       break;
     case 840:
       vmul_840(a,b,p,N);
       vsum_840(a,b,s,N);
       break;
     case 841:
       vmul_841(a,b,p,N);
       vsum_841(a,b,s,N);
       break;
     case 842:
       vmul_842(a,b,p,N);
       vsum_842(a,b,s,N);
       break;
     case 843:
       vmul_843(a,b,p,N);
       vsum_843(a,b,s,N);
       break;
     case 844:
       vmul_844(a,b,p,N);
       vsum_844(a,b,s,N);
       break;
     case 845:
       vmul_845(a,b,p,N);
       vsum_845(a,b,s,N);
       break;
     case 846:
       vmul_846(a,b,p,N);
       vsum_846(a,b,s,N);
       break;
     case 847:
       vmul_847(a,b,p,N);
       vsum_847(a,b,s,N);
       break;
     case 848:
       vmul_848(a,b,p,N);
       vsum_848(a,b,s,N);
       break;
     case 849:
       vmul_849(a,b,p,N);
       vsum_849(a,b,s,N);
       break;
     case 850:
       vmul_850(a,b,p,N);
       vsum_850(a,b,s,N);
       break;
     case 851:
       vmul_851(a,b,p,N);
       vsum_851(a,b,s,N);
       break;
     case 852:
       vmul_852(a,b,p,N);
       vsum_852(a,b,s,N);
       break;
     case 853:
       vmul_853(a,b,p,N);
       vsum_853(a,b,s,N);
       break;
     case 854:
       vmul_854(a,b,p,N);
       vsum_854(a,b,s,N);
       break;
     case 855:
       vmul_855(a,b,p,N);
       vsum_855(a,b,s,N);
       break;
     case 856:
       vmul_856(a,b,p,N);
       vsum_856(a,b,s,N);
       break;
     case 857:
       vmul_857(a,b,p,N);
       vsum_857(a,b,s,N);
       break;
     case 858:
       vmul_858(a,b,p,N);
       vsum_858(a,b,s,N);
       break;
     case 859:
       vmul_859(a,b,p,N);
       vsum_859(a,b,s,N);
       break;
     case 860:
       vmul_860(a,b,p,N);
       vsum_860(a,b,s,N);
       break;
     case 861:
       vmul_861(a,b,p,N);
       vsum_861(a,b,s,N);
       break;
     case 862:
       vmul_862(a,b,p,N);
       vsum_862(a,b,s,N);
       break;
     case 863:
       vmul_863(a,b,p,N);
       vsum_863(a,b,s,N);
       break;
     case 864:
       vmul_864(a,b,p,N);
       vsum_864(a,b,s,N);
       break;
     case 865:
       vmul_865(a,b,p,N);
       vsum_865(a,b,s,N);
       break;
     case 866:
       vmul_866(a,b,p,N);
       vsum_866(a,b,s,N);
       break;
     case 867:
       vmul_867(a,b,p,N);
       vsum_867(a,b,s,N);
       break;
     case 868:
       vmul_868(a,b,p,N);
       vsum_868(a,b,s,N);
       break;
     case 869:
       vmul_869(a,b,p,N);
       vsum_869(a,b,s,N);
       break;
     case 870:
       vmul_870(a,b,p,N);
       vsum_870(a,b,s,N);
       break;
     case 871:
       vmul_871(a,b,p,N);
       vsum_871(a,b,s,N);
       break;
     case 872:
       vmul_872(a,b,p,N);
       vsum_872(a,b,s,N);
       break;
     case 873:
       vmul_873(a,b,p,N);
       vsum_873(a,b,s,N);
       break;
     case 874:
       vmul_874(a,b,p,N);
       vsum_874(a,b,s,N);
       break;
     case 875:
       vmul_875(a,b,p,N);
       vsum_875(a,b,s,N);
       break;
     case 876:
       vmul_876(a,b,p,N);
       vsum_876(a,b,s,N);
       break;
     case 877:
       vmul_877(a,b,p,N);
       vsum_877(a,b,s,N);
       break;
     case 878:
       vmul_878(a,b,p,N);
       vsum_878(a,b,s,N);
       break;
     case 879:
       vmul_879(a,b,p,N);
       vsum_879(a,b,s,N);
       break;
     case 880:
       vmul_880(a,b,p,N);
       vsum_880(a,b,s,N);
       break;
     case 881:
       vmul_881(a,b,p,N);
       vsum_881(a,b,s,N);
       break;
     case 882:
       vmul_882(a,b,p,N);
       vsum_882(a,b,s,N);
       break;
     case 883:
       vmul_883(a,b,p,N);
       vsum_883(a,b,s,N);
       break;
     case 884:
       vmul_884(a,b,p,N);
       vsum_884(a,b,s,N);
       break;
     case 885:
       vmul_885(a,b,p,N);
       vsum_885(a,b,s,N);
       break;
     case 886:
       vmul_886(a,b,p,N);
       vsum_886(a,b,s,N);
       break;
     case 887:
       vmul_887(a,b,p,N);
       vsum_887(a,b,s,N);
       break;
     case 888:
       vmul_888(a,b,p,N);
       vsum_888(a,b,s,N);
       break;
     case 889:
       vmul_889(a,b,p,N);
       vsum_889(a,b,s,N);
       break;
     case 890:
       vmul_890(a,b,p,N);
       vsum_890(a,b,s,N);
       break;
     case 891:
       vmul_891(a,b,p,N);
       vsum_891(a,b,s,N);
       break;
     case 892:
       vmul_892(a,b,p,N);
       vsum_892(a,b,s,N);
       break;
     case 893:
       vmul_893(a,b,p,N);
       vsum_893(a,b,s,N);
       break;
     case 894:
       vmul_894(a,b,p,N);
       vsum_894(a,b,s,N);
       break;
     case 895:
       vmul_895(a,b,p,N);
       vsum_895(a,b,s,N);
       break;
     case 896:
       vmul_896(a,b,p,N);
       vsum_896(a,b,s,N);
       break;
     case 897:
       vmul_897(a,b,p,N);
       vsum_897(a,b,s,N);
       break;
     case 898:
       vmul_898(a,b,p,N);
       vsum_898(a,b,s,N);
       break;
     case 899:
       vmul_899(a,b,p,N);
       vsum_899(a,b,s,N);
       break;
     case 900:
       vmul_900(a,b,p,N);
       vsum_900(a,b,s,N);
       break;
     case 901:
       vmul_901(a,b,p,N);
       vsum_901(a,b,s,N);
       break;
     case 902:
       vmul_902(a,b,p,N);
       vsum_902(a,b,s,N);
       break;
     case 903:
       vmul_903(a,b,p,N);
       vsum_903(a,b,s,N);
       break;
     case 904:
       vmul_904(a,b,p,N);
       vsum_904(a,b,s,N);
       break;
     case 905:
       vmul_905(a,b,p,N);
       vsum_905(a,b,s,N);
       break;
     case 906:
       vmul_906(a,b,p,N);
       vsum_906(a,b,s,N);
       break;
     case 907:
       vmul_907(a,b,p,N);
       vsum_907(a,b,s,N);
       break;
     case 908:
       vmul_908(a,b,p,N);
       vsum_908(a,b,s,N);
       break;
     case 909:
       vmul_909(a,b,p,N);
       vsum_909(a,b,s,N);
       break;
     case 910:
       vmul_910(a,b,p,N);
       vsum_910(a,b,s,N);
       break;
     case 911:
       vmul_911(a,b,p,N);
       vsum_911(a,b,s,N);
       break;
     case 912:
       vmul_912(a,b,p,N);
       vsum_912(a,b,s,N);
       break;
     case 913:
       vmul_913(a,b,p,N);
       vsum_913(a,b,s,N);
       break;
     case 914:
       vmul_914(a,b,p,N);
       vsum_914(a,b,s,N);
       break;
     case 915:
       vmul_915(a,b,p,N);
       vsum_915(a,b,s,N);
       break;
     case 916:
       vmul_916(a,b,p,N);
       vsum_916(a,b,s,N);
       break;
     case 917:
       vmul_917(a,b,p,N);
       vsum_917(a,b,s,N);
       break;
     case 918:
       vmul_918(a,b,p,N);
       vsum_918(a,b,s,N);
       break;
     case 919:
       vmul_919(a,b,p,N);
       vsum_919(a,b,s,N);
       break;
     case 920:
       vmul_920(a,b,p,N);
       vsum_920(a,b,s,N);
       break;
     case 921:
       vmul_921(a,b,p,N);
       vsum_921(a,b,s,N);
       break;
     case 922:
       vmul_922(a,b,p,N);
       vsum_922(a,b,s,N);
       break;
     case 923:
       vmul_923(a,b,p,N);
       vsum_923(a,b,s,N);
       break;
     case 924:
       vmul_924(a,b,p,N);
       vsum_924(a,b,s,N);
       break;
     case 925:
       vmul_925(a,b,p,N);
       vsum_925(a,b,s,N);
       break;
     case 926:
       vmul_926(a,b,p,N);
       vsum_926(a,b,s,N);
       break;
     case 927:
       vmul_927(a,b,p,N);
       vsum_927(a,b,s,N);
       break;
     case 928:
       vmul_928(a,b,p,N);
       vsum_928(a,b,s,N);
       break;
     case 929:
       vmul_929(a,b,p,N);
       vsum_929(a,b,s,N);
       break;
     case 930:
       vmul_930(a,b,p,N);
       vsum_930(a,b,s,N);
       break;
     case 931:
       vmul_931(a,b,p,N);
       vsum_931(a,b,s,N);
       break;
     case 932:
       vmul_932(a,b,p,N);
       vsum_932(a,b,s,N);
       break;
     case 933:
       vmul_933(a,b,p,N);
       vsum_933(a,b,s,N);
       break;
     case 934:
       vmul_934(a,b,p,N);
       vsum_934(a,b,s,N);
       break;
     case 935:
       vmul_935(a,b,p,N);
       vsum_935(a,b,s,N);
       break;
     case 936:
       vmul_936(a,b,p,N);
       vsum_936(a,b,s,N);
       break;
     case 937:
       vmul_937(a,b,p,N);
       vsum_937(a,b,s,N);
       break;
     case 938:
       vmul_938(a,b,p,N);
       vsum_938(a,b,s,N);
       break;
     case 939:
       vmul_939(a,b,p,N);
       vsum_939(a,b,s,N);
       break;
     case 940:
       vmul_940(a,b,p,N);
       vsum_940(a,b,s,N);
       break;
     case 941:
       vmul_941(a,b,p,N);
       vsum_941(a,b,s,N);
       break;
     case 942:
       vmul_942(a,b,p,N);
       vsum_942(a,b,s,N);
       break;
     case 943:
       vmul_943(a,b,p,N);
       vsum_943(a,b,s,N);
       break;
     case 944:
       vmul_944(a,b,p,N);
       vsum_944(a,b,s,N);
       break;
     case 945:
       vmul_945(a,b,p,N);
       vsum_945(a,b,s,N);
       break;
     case 946:
       vmul_946(a,b,p,N);
       vsum_946(a,b,s,N);
       break;
     case 947:
       vmul_947(a,b,p,N);
       vsum_947(a,b,s,N);
       break;
     case 948:
       vmul_948(a,b,p,N);
       vsum_948(a,b,s,N);
       break;
     case 949:
       vmul_949(a,b,p,N);
       vsum_949(a,b,s,N);
       break;
     case 950:
       vmul_950(a,b,p,N);
       vsum_950(a,b,s,N);
       break;
     case 951:
       vmul_951(a,b,p,N);
       vsum_951(a,b,s,N);
       break;
     case 952:
       vmul_952(a,b,p,N);
       vsum_952(a,b,s,N);
       break;
     case 953:
       vmul_953(a,b,p,N);
       vsum_953(a,b,s,N);
       break;
     case 954:
       vmul_954(a,b,p,N);
       vsum_954(a,b,s,N);
       break;
     case 955:
       vmul_955(a,b,p,N);
       vsum_955(a,b,s,N);
       break;
     case 956:
       vmul_956(a,b,p,N);
       vsum_956(a,b,s,N);
       break;
     case 957:
       vmul_957(a,b,p,N);
       vsum_957(a,b,s,N);
       break;
     case 958:
       vmul_958(a,b,p,N);
       vsum_958(a,b,s,N);
       break;
     case 959:
       vmul_959(a,b,p,N);
       vsum_959(a,b,s,N);
       break;
     case 960:
       vmul_960(a,b,p,N);
       vsum_960(a,b,s,N);
       break;
     case 961:
       vmul_961(a,b,p,N);
       vsum_961(a,b,s,N);
       break;
     case 962:
       vmul_962(a,b,p,N);
       vsum_962(a,b,s,N);
       break;
     case 963:
       vmul_963(a,b,p,N);
       vsum_963(a,b,s,N);
       break;
     case 964:
       vmul_964(a,b,p,N);
       vsum_964(a,b,s,N);
       break;
     case 965:
       vmul_965(a,b,p,N);
       vsum_965(a,b,s,N);
       break;
     case 966:
       vmul_966(a,b,p,N);
       vsum_966(a,b,s,N);
       break;
     case 967:
       vmul_967(a,b,p,N);
       vsum_967(a,b,s,N);
       break;
     case 968:
       vmul_968(a,b,p,N);
       vsum_968(a,b,s,N);
       break;
     case 969:
       vmul_969(a,b,p,N);
       vsum_969(a,b,s,N);
       break;
     case 970:
       vmul_970(a,b,p,N);
       vsum_970(a,b,s,N);
       break;
     case 971:
       vmul_971(a,b,p,N);
       vsum_971(a,b,s,N);
       break;
     case 972:
       vmul_972(a,b,p,N);
       vsum_972(a,b,s,N);
       break;
     case 973:
       vmul_973(a,b,p,N);
       vsum_973(a,b,s,N);
       break;
     case 974:
       vmul_974(a,b,p,N);
       vsum_974(a,b,s,N);
       break;
     case 975:
       vmul_975(a,b,p,N);
       vsum_975(a,b,s,N);
       break;
     case 976:
       vmul_976(a,b,p,N);
       vsum_976(a,b,s,N);
       break;
     case 977:
       vmul_977(a,b,p,N);
       vsum_977(a,b,s,N);
       break;
     case 978:
       vmul_978(a,b,p,N);
       vsum_978(a,b,s,N);
       break;
     case 979:
       vmul_979(a,b,p,N);
       vsum_979(a,b,s,N);
       break;
     case 980:
       vmul_980(a,b,p,N);
       vsum_980(a,b,s,N);
       break;
     case 981:
       vmul_981(a,b,p,N);
       vsum_981(a,b,s,N);
       break;
     case 982:
       vmul_982(a,b,p,N);
       vsum_982(a,b,s,N);
       break;
     case 983:
       vmul_983(a,b,p,N);
       vsum_983(a,b,s,N);
       break;
     case 984:
       vmul_984(a,b,p,N);
       vsum_984(a,b,s,N);
       break;
     case 985:
       vmul_985(a,b,p,N);
       vsum_985(a,b,s,N);
       break;
     case 986:
       vmul_986(a,b,p,N);
       vsum_986(a,b,s,N);
       break;
     case 987:
       vmul_987(a,b,p,N);
       vsum_987(a,b,s,N);
       break;
     case 988:
       vmul_988(a,b,p,N);
       vsum_988(a,b,s,N);
       break;
     case 989:
       vmul_989(a,b,p,N);
       vsum_989(a,b,s,N);
       break;
     case 990:
       vmul_990(a,b,p,N);
       vsum_990(a,b,s,N);
       break;
     case 991:
       vmul_991(a,b,p,N);
       vsum_991(a,b,s,N);
       break;
     case 992:
       vmul_992(a,b,p,N);
       vsum_992(a,b,s,N);
       break;
     case 993:
       vmul_993(a,b,p,N);
       vsum_993(a,b,s,N);
       break;
     case 994:
       vmul_994(a,b,p,N);
       vsum_994(a,b,s,N);
       break;
     case 995:
       vmul_995(a,b,p,N);
       vsum_995(a,b,s,N);
       break;
     case 996:
       vmul_996(a,b,p,N);
       vsum_996(a,b,s,N);
       break;
     case 997:
       vmul_997(a,b,p,N);
       vsum_997(a,b,s,N);
       break;
     case 998:
       vmul_998(a,b,p,N);
       vsum_998(a,b,s,N);
       break;
     case 999:
       vmul_999(a,b,p,N);
       vsum_999(a,b,s,N);
       break;
   }

   // check the results
   for(int i=0;i<N;i++) 
      if((p[i]!=pcheck[i])|(s[i]!=scheck[i])) flag=i;

   if (flag != -1) {
      printf("Fail p[%d]=%d   pcheck[%d]=%d\n",
         flag,p[flag],flag,pcheck[flag]);
      printf("Fail s[%d]=%d   scheck[%d]=%d\n",
         flag,s[flag],flag,scheck[flag]);
      return 1;
   } else {
      printf("Success\n");
      return 0;
   }
}
