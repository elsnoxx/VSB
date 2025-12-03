package cz.transys.moldapp.buisines.apicalls.moldmount

import kotlinx.serialization.Serializable


@Serializable
data class MoldMountResquest(
    val carrirer_no: String,
    val carrirer_name: String,
    val car_code1: String,
    val car_code2: String,
    val mold_code1: String,
    val mold_code2: String,
    val type1: String,
    val type2: String
)