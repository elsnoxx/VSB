package cz.transys.moldapp.ui.apicalls



class MoldRepairRepository {

    suspend fun getAllRepairTypes(): List<RepairTypes> {
        return ApiClient.get("foampad/moldpda/repair/types")
    }

    suspend fun getMoldRepairInfo(moldCode: String): MoldData {
        return ApiClient.get("foampad/moldpda/repair/$moldCode")
    }

    suspend fun postMoldRepair(moldRepair: MoldRepairSent): Boolean {
        return ApiClient.post("foampad/moldpda/repair", moldRepair)
    }
}

data class RepairTypes(
    val repair_code: String,
    val repair_name: String
)

data class MoldData(
    val car_code: String,
    val mold_code: String,
    val mold_name: String,
    val save_dttm: String
)

data class MoldRepairSent(
    val sysId: String,
    val moldCode: String,
    val repairCode: String,
    val empId: String
)