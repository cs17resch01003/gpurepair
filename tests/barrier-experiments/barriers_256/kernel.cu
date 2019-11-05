//pass
//--blockDim=32 --gridDim=1

#include <cuda.h>

__global__ void race (int* __restrict__ A1, int* __restrict__ A2, int* __restrict__ A3, int* __restrict__ A4, int* __restrict__ A5, int* __restrict__ A6, int* __restrict__ A7, int* __restrict__ A8, int* __restrict__ A9, int* __restrict__ A10, int* __restrict__ A11, int* __restrict__ A12, int* __restrict__ A13, int* __restrict__ A14, int* __restrict__ A15, int* __restrict__ A16, int* __restrict__ A17, int* __restrict__ A18, int* __restrict__ A19, int* __restrict__ A20, int* __restrict__ A21, int* __restrict__ A22, int* __restrict__ A23, int* __restrict__ A24, int* __restrict__ A25, int* __restrict__ A26, int* __restrict__ A27, int* __restrict__ A28, int* __restrict__ A29, int* __restrict__ A30, int* __restrict__ A31, int* __restrict__ A32, int* __restrict__ A33, int* __restrict__ A34, int* __restrict__ A35, int* __restrict__ A36, int* __restrict__ A37, int* __restrict__ A38, int* __restrict__ A39, int* __restrict__ A40, int* __restrict__ A41, int* __restrict__ A42, int* __restrict__ A43, int* __restrict__ A44, int* __restrict__ A45, int* __restrict__ A46, int* __restrict__ A47, int* __restrict__ A48, int* __restrict__ A49, int* __restrict__ A50, int* __restrict__ A51, int* __restrict__ A52, int* __restrict__ A53, int* __restrict__ A54, int* __restrict__ A55, int* __restrict__ A56, int* __restrict__ A57, int* __restrict__ A58, int* __restrict__ A59, int* __restrict__ A60, int* __restrict__ A61, int* __restrict__ A62, int* __restrict__ A63, int* __restrict__ A64, int* __restrict__ A65, int* __restrict__ A66, int* __restrict__ A67, int* __restrict__ A68, int* __restrict__ A69, int* __restrict__ A70, int* __restrict__ A71, int* __restrict__ A72, int* __restrict__ A73, int* __restrict__ A74, int* __restrict__ A75, int* __restrict__ A76, int* __restrict__ A77, int* __restrict__ A78, int* __restrict__ A79, int* __restrict__ A80, int* __restrict__ A81, int* __restrict__ A82, int* __restrict__ A83, int* __restrict__ A84, int* __restrict__ A85, int* __restrict__ A86, int* __restrict__ A87, int* __restrict__ A88, int* __restrict__ A89, int* __restrict__ A90, int* __restrict__ A91, int* __restrict__ A92, int* __restrict__ A93, int* __restrict__ A94, int* __restrict__ A95, int* __restrict__ A96, int* __restrict__ A97, int* __restrict__ A98, int* __restrict__ A99, int* __restrict__ A100, int* __restrict__ A101, int* __restrict__ A102, int* __restrict__ A103, int* __restrict__ A104, int* __restrict__ A105, int* __restrict__ A106, int* __restrict__ A107, int* __restrict__ A108, int* __restrict__ A109, int* __restrict__ A110, int* __restrict__ A111, int* __restrict__ A112, int* __restrict__ A113, int* __restrict__ A114, int* __restrict__ A115, int* __restrict__ A116, int* __restrict__ A117, int* __restrict__ A118, int* __restrict__ A119, int* __restrict__ A120, int* __restrict__ A121, int* __restrict__ A122, int* __restrict__ A123, int* __restrict__ A124, int* __restrict__ A125, int* __restrict__ A126, int* __restrict__ A127, int* __restrict__ A128, int* __restrict__ A129, int* __restrict__ A130, int* __restrict__ A131, int* __restrict__ A132, int* __restrict__ A133, int* __restrict__ A134, int* __restrict__ A135, int* __restrict__ A136, int* __restrict__ A137, int* __restrict__ A138, int* __restrict__ A139, int* __restrict__ A140, int* __restrict__ A141, int* __restrict__ A142, int* __restrict__ A143, int* __restrict__ A144, int* __restrict__ A145, int* __restrict__ A146, int* __restrict__ A147, int* __restrict__ A148, int* __restrict__ A149, int* __restrict__ A150, int* __restrict__ A151, int* __restrict__ A152, int* __restrict__ A153, int* __restrict__ A154, int* __restrict__ A155, int* __restrict__ A156, int* __restrict__ A157, int* __restrict__ A158, int* __restrict__ A159, int* __restrict__ A160, int* __restrict__ A161, int* __restrict__ A162, int* __restrict__ A163, int* __restrict__ A164, int* __restrict__ A165, int* __restrict__ A166, int* __restrict__ A167, int* __restrict__ A168, int* __restrict__ A169, int* __restrict__ A170, int* __restrict__ A171, int* __restrict__ A172, int* __restrict__ A173, int* __restrict__ A174, int* __restrict__ A175, int* __restrict__ A176, int* __restrict__ A177, int* __restrict__ A178, int* __restrict__ A179, int* __restrict__ A180, int* __restrict__ A181, int* __restrict__ A182, int* __restrict__ A183, int* __restrict__ A184, int* __restrict__ A185, int* __restrict__ A186, int* __restrict__ A187, int* __restrict__ A188, int* __restrict__ A189, int* __restrict__ A190, int* __restrict__ A191, int* __restrict__ A192, int* __restrict__ A193, int* __restrict__ A194, int* __restrict__ A195, int* __restrict__ A196, int* __restrict__ A197, int* __restrict__ A198, int* __restrict__ A199, int* __restrict__ A200, int* __restrict__ A201, int* __restrict__ A202, int* __restrict__ A203, int* __restrict__ A204, int* __restrict__ A205, int* __restrict__ A206, int* __restrict__ A207, int* __restrict__ A208, int* __restrict__ A209, int* __restrict__ A210, int* __restrict__ A211, int* __restrict__ A212, int* __restrict__ A213, int* __restrict__ A214, int* __restrict__ A215, int* __restrict__ A216, int* __restrict__ A217, int* __restrict__ A218, int* __restrict__ A219, int* __restrict__ A220, int* __restrict__ A221, int* __restrict__ A222, int* __restrict__ A223, int* __restrict__ A224, int* __restrict__ A225, int* __restrict__ A226, int* __restrict__ A227, int* __restrict__ A228, int* __restrict__ A229, int* __restrict__ A230, int* __restrict__ A231, int* __restrict__ A232, int* __restrict__ A233, int* __restrict__ A234, int* __restrict__ A235, int* __restrict__ A236, int* __restrict__ A237, int* __restrict__ A238, int* __restrict__ A239, int* __restrict__ A240, int* __restrict__ A241, int* __restrict__ A242, int* __restrict__ A243, int* __restrict__ A244, int* __restrict__ A245, int* __restrict__ A246, int* __restrict__ A247, int* __restrict__ A248, int* __restrict__ A249, int* __restrict__ A250, int* __restrict__ A251, int* __restrict__ A252, int* __restrict__ A253, int* __restrict__ A254, int* __restrict__ A255, int* __restrict__ A256)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;
  int temp = 0;

  temp = A1[idx + 1];
  A1[idx] = temp;

  temp = A2[idx + 1];
  A2[idx] = temp;

  temp = A3[idx + 1];
  A3[idx] = temp;

  temp = A4[idx + 1];
  A4[idx] = temp;

  temp = A5[idx + 1];
  A5[idx] = temp;

  temp = A6[idx + 1];
  A6[idx] = temp;

  temp = A7[idx + 1];
  A7[idx] = temp;

  temp = A8[idx + 1];
  A8[idx] = temp;

  temp = A9[idx + 1];
  A9[idx] = temp;

  temp = A10[idx + 1];
  A10[idx] = temp;

  temp = A11[idx + 1];
  A11[idx] = temp;

  temp = A12[idx + 1];
  A12[idx] = temp;

  temp = A13[idx + 1];
  A13[idx] = temp;

  temp = A14[idx + 1];
  A14[idx] = temp;

  temp = A15[idx + 1];
  A15[idx] = temp;

  temp = A16[idx + 1];
  A16[idx] = temp;

  temp = A17[idx + 1];
  A17[idx] = temp;

  temp = A18[idx + 1];
  A18[idx] = temp;

  temp = A19[idx + 1];
  A19[idx] = temp;

  temp = A20[idx + 1];
  A20[idx] = temp;

  temp = A21[idx + 1];
  A21[idx] = temp;

  temp = A22[idx + 1];
  A22[idx] = temp;

  temp = A23[idx + 1];
  A23[idx] = temp;

  temp = A24[idx + 1];
  A24[idx] = temp;

  temp = A25[idx + 1];
  A25[idx] = temp;

  temp = A26[idx + 1];
  A26[idx] = temp;

  temp = A27[idx + 1];
  A27[idx] = temp;

  temp = A28[idx + 1];
  A28[idx] = temp;

  temp = A29[idx + 1];
  A29[idx] = temp;

  temp = A30[idx + 1];
  A30[idx] = temp;

  temp = A31[idx + 1];
  A31[idx] = temp;

  temp = A32[idx + 1];
  A32[idx] = temp;

  temp = A33[idx + 1];
  A33[idx] = temp;

  temp = A34[idx + 1];
  A34[idx] = temp;

  temp = A35[idx + 1];
  A35[idx] = temp;

  temp = A36[idx + 1];
  A36[idx] = temp;

  temp = A37[idx + 1];
  A37[idx] = temp;

  temp = A38[idx + 1];
  A38[idx] = temp;

  temp = A39[idx + 1];
  A39[idx] = temp;

  temp = A40[idx + 1];
  A40[idx] = temp;

  temp = A41[idx + 1];
  A41[idx] = temp;

  temp = A42[idx + 1];
  A42[idx] = temp;

  temp = A43[idx + 1];
  A43[idx] = temp;

  temp = A44[idx + 1];
  A44[idx] = temp;

  temp = A45[idx + 1];
  A45[idx] = temp;

  temp = A46[idx + 1];
  A46[idx] = temp;

  temp = A47[idx + 1];
  A47[idx] = temp;

  temp = A48[idx + 1];
  A48[idx] = temp;

  temp = A49[idx + 1];
  A49[idx] = temp;

  temp = A50[idx + 1];
  A50[idx] = temp;

  temp = A51[idx + 1];
  A51[idx] = temp;

  temp = A52[idx + 1];
  A52[idx] = temp;

  temp = A53[idx + 1];
  A53[idx] = temp;

  temp = A54[idx + 1];
  A54[idx] = temp;

  temp = A55[idx + 1];
  A55[idx] = temp;

  temp = A56[idx + 1];
  A56[idx] = temp;

  temp = A57[idx + 1];
  A57[idx] = temp;

  temp = A58[idx + 1];
  A58[idx] = temp;

  temp = A59[idx + 1];
  A59[idx] = temp;

  temp = A60[idx + 1];
  A60[idx] = temp;

  temp = A61[idx + 1];
  A61[idx] = temp;

  temp = A62[idx + 1];
  A62[idx] = temp;

  temp = A63[idx + 1];
  A63[idx] = temp;

  temp = A64[idx + 1];
  A64[idx] = temp;

  temp = A65[idx + 1];
  A65[idx] = temp;

  temp = A66[idx + 1];
  A66[idx] = temp;

  temp = A67[idx + 1];
  A67[idx] = temp;

  temp = A68[idx + 1];
  A68[idx] = temp;

  temp = A69[idx + 1];
  A69[idx] = temp;

  temp = A70[idx + 1];
  A70[idx] = temp;

  temp = A71[idx + 1];
  A71[idx] = temp;

  temp = A72[idx + 1];
  A72[idx] = temp;

  temp = A73[idx + 1];
  A73[idx] = temp;

  temp = A74[idx + 1];
  A74[idx] = temp;

  temp = A75[idx + 1];
  A75[idx] = temp;

  temp = A76[idx + 1];
  A76[idx] = temp;

  temp = A77[idx + 1];
  A77[idx] = temp;

  temp = A78[idx + 1];
  A78[idx] = temp;

  temp = A79[idx + 1];
  A79[idx] = temp;

  temp = A80[idx + 1];
  A80[idx] = temp;

  temp = A81[idx + 1];
  A81[idx] = temp;

  temp = A82[idx + 1];
  A82[idx] = temp;

  temp = A83[idx + 1];
  A83[idx] = temp;

  temp = A84[idx + 1];
  A84[idx] = temp;

  temp = A85[idx + 1];
  A85[idx] = temp;

  temp = A86[idx + 1];
  A86[idx] = temp;

  temp = A87[idx + 1];
  A87[idx] = temp;

  temp = A88[idx + 1];
  A88[idx] = temp;

  temp = A89[idx + 1];
  A89[idx] = temp;

  temp = A90[idx + 1];
  A90[idx] = temp;

  temp = A91[idx + 1];
  A91[idx] = temp;

  temp = A92[idx + 1];
  A92[idx] = temp;

  temp = A93[idx + 1];
  A93[idx] = temp;

  temp = A94[idx + 1];
  A94[idx] = temp;

  temp = A95[idx + 1];
  A95[idx] = temp;

  temp = A96[idx + 1];
  A96[idx] = temp;

  temp = A97[idx + 1];
  A97[idx] = temp;

  temp = A98[idx + 1];
  A98[idx] = temp;

  temp = A99[idx + 1];
  A99[idx] = temp;

  temp = A100[idx + 1];
  A100[idx] = temp;

  temp = A101[idx + 1];
  A101[idx] = temp;

  temp = A102[idx + 1];
  A102[idx] = temp;

  temp = A103[idx + 1];
  A103[idx] = temp;

  temp = A104[idx + 1];
  A104[idx] = temp;

  temp = A105[idx + 1];
  A105[idx] = temp;

  temp = A106[idx + 1];
  A106[idx] = temp;

  temp = A107[idx + 1];
  A107[idx] = temp;

  temp = A108[idx + 1];
  A108[idx] = temp;

  temp = A109[idx + 1];
  A109[idx] = temp;

  temp = A110[idx + 1];
  A110[idx] = temp;

  temp = A111[idx + 1];
  A111[idx] = temp;

  temp = A112[idx + 1];
  A112[idx] = temp;

  temp = A113[idx + 1];
  A113[idx] = temp;

  temp = A114[idx + 1];
  A114[idx] = temp;

  temp = A115[idx + 1];
  A115[idx] = temp;

  temp = A116[idx + 1];
  A116[idx] = temp;

  temp = A117[idx + 1];
  A117[idx] = temp;

  temp = A118[idx + 1];
  A118[idx] = temp;

  temp = A119[idx + 1];
  A119[idx] = temp;

  temp = A120[idx + 1];
  A120[idx] = temp;

  temp = A121[idx + 1];
  A121[idx] = temp;

  temp = A122[idx + 1];
  A122[idx] = temp;

  temp = A123[idx + 1];
  A123[idx] = temp;

  temp = A124[idx + 1];
  A124[idx] = temp;

  temp = A125[idx + 1];
  A125[idx] = temp;

  temp = A126[idx + 1];
  A126[idx] = temp;

  temp = A127[idx + 1];
  A127[idx] = temp;

  temp = A128[idx + 1];
  A128[idx] = temp;

  temp = A129[idx + 1];
  A129[idx] = temp;

  temp = A130[idx + 1];
  A130[idx] = temp;

  temp = A131[idx + 1];
  A131[idx] = temp;

  temp = A132[idx + 1];
  A132[idx] = temp;

  temp = A133[idx + 1];
  A133[idx] = temp;

  temp = A134[idx + 1];
  A134[idx] = temp;

  temp = A135[idx + 1];
  A135[idx] = temp;

  temp = A136[idx + 1];
  A136[idx] = temp;

  temp = A137[idx + 1];
  A137[idx] = temp;

  temp = A138[idx + 1];
  A138[idx] = temp;

  temp = A139[idx + 1];
  A139[idx] = temp;

  temp = A140[idx + 1];
  A140[idx] = temp;

  temp = A141[idx + 1];
  A141[idx] = temp;

  temp = A142[idx + 1];
  A142[idx] = temp;

  temp = A143[idx + 1];
  A143[idx] = temp;

  temp = A144[idx + 1];
  A144[idx] = temp;

  temp = A145[idx + 1];
  A145[idx] = temp;

  temp = A146[idx + 1];
  A146[idx] = temp;

  temp = A147[idx + 1];
  A147[idx] = temp;

  temp = A148[idx + 1];
  A148[idx] = temp;

  temp = A149[idx + 1];
  A149[idx] = temp;

  temp = A150[idx + 1];
  A150[idx] = temp;

  temp = A151[idx + 1];
  A151[idx] = temp;

  temp = A152[idx + 1];
  A152[idx] = temp;

  temp = A153[idx + 1];
  A153[idx] = temp;

  temp = A154[idx + 1];
  A154[idx] = temp;

  temp = A155[idx + 1];
  A155[idx] = temp;

  temp = A156[idx + 1];
  A156[idx] = temp;

  temp = A157[idx + 1];
  A157[idx] = temp;

  temp = A158[idx + 1];
  A158[idx] = temp;

  temp = A159[idx + 1];
  A159[idx] = temp;

  temp = A160[idx + 1];
  A160[idx] = temp;

  temp = A161[idx + 1];
  A161[idx] = temp;

  temp = A162[idx + 1];
  A162[idx] = temp;

  temp = A163[idx + 1];
  A163[idx] = temp;

  temp = A164[idx + 1];
  A164[idx] = temp;

  temp = A165[idx + 1];
  A165[idx] = temp;

  temp = A166[idx + 1];
  A166[idx] = temp;

  temp = A167[idx + 1];
  A167[idx] = temp;

  temp = A168[idx + 1];
  A168[idx] = temp;

  temp = A169[idx + 1];
  A169[idx] = temp;

  temp = A170[idx + 1];
  A170[idx] = temp;

  temp = A171[idx + 1];
  A171[idx] = temp;

  temp = A172[idx + 1];
  A172[idx] = temp;

  temp = A173[idx + 1];
  A173[idx] = temp;

  temp = A174[idx + 1];
  A174[idx] = temp;

  temp = A175[idx + 1];
  A175[idx] = temp;

  temp = A176[idx + 1];
  A176[idx] = temp;

  temp = A177[idx + 1];
  A177[idx] = temp;

  temp = A178[idx + 1];
  A178[idx] = temp;

  temp = A179[idx + 1];
  A179[idx] = temp;

  temp = A180[idx + 1];
  A180[idx] = temp;

  temp = A181[idx + 1];
  A181[idx] = temp;

  temp = A182[idx + 1];
  A182[idx] = temp;

  temp = A183[idx + 1];
  A183[idx] = temp;

  temp = A184[idx + 1];
  A184[idx] = temp;

  temp = A185[idx + 1];
  A185[idx] = temp;

  temp = A186[idx + 1];
  A186[idx] = temp;

  temp = A187[idx + 1];
  A187[idx] = temp;

  temp = A188[idx + 1];
  A188[idx] = temp;

  temp = A189[idx + 1];
  A189[idx] = temp;

  temp = A190[idx + 1];
  A190[idx] = temp;

  temp = A191[idx + 1];
  A191[idx] = temp;

  temp = A192[idx + 1];
  A192[idx] = temp;

  temp = A193[idx + 1];
  A193[idx] = temp;

  temp = A194[idx + 1];
  A194[idx] = temp;

  temp = A195[idx + 1];
  A195[idx] = temp;

  temp = A196[idx + 1];
  A196[idx] = temp;

  temp = A197[idx + 1];
  A197[idx] = temp;

  temp = A198[idx + 1];
  A198[idx] = temp;

  temp = A199[idx + 1];
  A199[idx] = temp;

  temp = A200[idx + 1];
  A200[idx] = temp;

  temp = A201[idx + 1];
  A201[idx] = temp;

  temp = A202[idx + 1];
  A202[idx] = temp;

  temp = A203[idx + 1];
  A203[idx] = temp;

  temp = A204[idx + 1];
  A204[idx] = temp;

  temp = A205[idx + 1];
  A205[idx] = temp;

  temp = A206[idx + 1];
  A206[idx] = temp;

  temp = A207[idx + 1];
  A207[idx] = temp;

  temp = A208[idx + 1];
  A208[idx] = temp;

  temp = A209[idx + 1];
  A209[idx] = temp;

  temp = A210[idx + 1];
  A210[idx] = temp;

  temp = A211[idx + 1];
  A211[idx] = temp;

  temp = A212[idx + 1];
  A212[idx] = temp;

  temp = A213[idx + 1];
  A213[idx] = temp;

  temp = A214[idx + 1];
  A214[idx] = temp;

  temp = A215[idx + 1];
  A215[idx] = temp;

  temp = A216[idx + 1];
  A216[idx] = temp;

  temp = A217[idx + 1];
  A217[idx] = temp;

  temp = A218[idx + 1];
  A218[idx] = temp;

  temp = A219[idx + 1];
  A219[idx] = temp;

  temp = A220[idx + 1];
  A220[idx] = temp;

  temp = A221[idx + 1];
  A221[idx] = temp;

  temp = A222[idx + 1];
  A222[idx] = temp;

  temp = A223[idx + 1];
  A223[idx] = temp;

  temp = A224[idx + 1];
  A224[idx] = temp;

  temp = A225[idx + 1];
  A225[idx] = temp;

  temp = A226[idx + 1];
  A226[idx] = temp;

  temp = A227[idx + 1];
  A227[idx] = temp;

  temp = A228[idx + 1];
  A228[idx] = temp;

  temp = A229[idx + 1];
  A229[idx] = temp;

  temp = A230[idx + 1];
  A230[idx] = temp;

  temp = A231[idx + 1];
  A231[idx] = temp;

  temp = A232[idx + 1];
  A232[idx] = temp;

  temp = A233[idx + 1];
  A233[idx] = temp;

  temp = A234[idx + 1];
  A234[idx] = temp;

  temp = A235[idx + 1];
  A235[idx] = temp;

  temp = A236[idx + 1];
  A236[idx] = temp;

  temp = A237[idx + 1];
  A237[idx] = temp;

  temp = A238[idx + 1];
  A238[idx] = temp;

  temp = A239[idx + 1];
  A239[idx] = temp;

  temp = A240[idx + 1];
  A240[idx] = temp;

  temp = A241[idx + 1];
  A241[idx] = temp;

  temp = A242[idx + 1];
  A242[idx] = temp;

  temp = A243[idx + 1];
  A243[idx] = temp;

  temp = A244[idx + 1];
  A244[idx] = temp;

  temp = A245[idx + 1];
  A245[idx] = temp;

  temp = A246[idx + 1];
  A246[idx] = temp;

  temp = A247[idx + 1];
  A247[idx] = temp;

  temp = A248[idx + 1];
  A248[idx] = temp;

  temp = A249[idx + 1];
  A249[idx] = temp;

  temp = A250[idx + 1];
  A250[idx] = temp;

  temp = A251[idx + 1];
  A251[idx] = temp;

  temp = A252[idx + 1];
  A252[idx] = temp;

  temp = A253[idx + 1];
  A253[idx] = temp;

  temp = A254[idx + 1];
  A254[idx] = temp;

  temp = A255[idx + 1];
  A255[idx] = temp;

  temp = A256[idx + 1];
  A256[idx] = temp;
}