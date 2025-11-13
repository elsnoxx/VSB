package cz.transys.moldapp.ui.apicalls



class MoldRepairRepository {

    suspend fun getAllRepairTypes(): List<RepairTypes> {
        return ApiClient.get("foampad/mold/repair/types")
    }

    suspend fun getMoldRepairInfo(moldCode: String): MoldData {
        return ApiClient.get("foampad/mold/repair/$moldCode")
    }

    suspend fun postMoldRepair(moldRepair: MoldRepairSent): Boolean {
        return ApiClient.post("foampad/mold/repair", moldRepair)
    }
}

data class RepairTypes(
    val repaiR_CODE: String,
    val repaiR_NAME2: String
)

data class MoldData(
    val caR_CODE: String,
    val molD_CODE: String,
    val molD_NAME: String,
    val savE_DTTM: String
)

data class MoldRepairSent(
    val sysId: String,
    val moldCode: String,
    val repairCode: String,
    val empId: String
)