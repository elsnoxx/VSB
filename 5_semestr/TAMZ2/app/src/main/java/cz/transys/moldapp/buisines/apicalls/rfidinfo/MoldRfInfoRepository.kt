package cz.transys.moldapp.buisines.apicalls.rfidinfo

import cz.transys.moldapp.buisines.apicalls.ApiClient

class MoldRfInfoRepository {
    suspend fun getTagInfo(tag: String): TagInfo? {
        return try {
            ApiClient.get("foampad/moldpda/taginfo?MOLD_CODE=$tag")
        } catch (e: Exception) {
            null
        }
    }
}


