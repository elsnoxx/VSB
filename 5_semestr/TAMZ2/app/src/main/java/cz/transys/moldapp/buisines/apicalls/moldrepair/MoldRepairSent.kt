package cz.transys.moldapp.buisines.apicalls.moldrepair

data class MoldRepairSent(
    val sysId: String,
    val moldCode: String,
    val repairCode: String,
    val empId: String
)
