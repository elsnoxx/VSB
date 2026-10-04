package cz.transys.moldapp.buisines.apicalls.rfidinfo

data class TagInfo(
    val mold_code: String,
    val mold_name: String,
    val car_code: String,
    val code_value1: String
)
