package cz.transys.moldapp.ui.apicalls

class MoldRfInfoRepository {
    suspend fun getTagInfo(tag: String): TagInfo {
        return ApiClient.get("foampad/mold/rftag/$tag")
    }
}

data class TagInfo(
    val mold: String,
    val type: String,
    val car: String,
    val status: String,
    val total: String
)

