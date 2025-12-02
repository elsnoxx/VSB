package cz.transys.moldapp.buisines.apicalls.moldrepair

import cz.transys.moldapp.buisines.apicalls.ApiClient
import cz.transys.moldapp.buisines.apicalls.moldapi.CarCodeList


class MoldRepairRepository {
    private var typeCache: List<RepairTypes>? = null
    suspend fun getAllRepairTypes(forceRefresh: Boolean = false): List<RepairTypes> {
        if (!forceRefresh && typeCache != null) {
            return typeCache!!
        }

        val result = ApiClient.get<List<RepairTypes>>("foampad/moldpda/repair/types")
        typeCache = result
        return result
    }

    suspend fun getMoldRepairInfo(moldCode: String): MoldData {
        return ApiClient.get("foampad/moldpda/repair/$moldCode")
    }

    suspend fun postMoldRepair(moldRepair: MoldRepairSent): Boolean {
        return ApiClient.post("foampad/moldpda/repair", moldRepair)
    }
}