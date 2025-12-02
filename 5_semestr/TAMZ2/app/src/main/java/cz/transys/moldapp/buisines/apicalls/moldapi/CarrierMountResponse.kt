package cz.transys.moldapp.buisines.apicalls.moldapi

data class CarrierMountResponse(
    val mold_code1: String?,
    val part_code1: String?,
    val mold_code2: String?,
    val part_code2: String?,
    val mold_name1: String?,
    val mold_name2: String?,
    val car_code1: String?,
    val car_code2: String?,
    val mut_part_flag: String?
)