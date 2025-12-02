package cz.transys.moldapp.buisines.apicalls.partchange

import cz.transys.moldapp.buisines.apicalls.ApiClient

class PartChangeRepository {
    suspend fun postPartChange(partChange: PartChangeRequest): Boolean {
        return ApiClient.post("foampad/moldpda/partchange", partChange)
    }
}