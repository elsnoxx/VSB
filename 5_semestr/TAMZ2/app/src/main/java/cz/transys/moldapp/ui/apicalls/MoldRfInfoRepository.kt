package cz.transys.moldapp.ui.apicalls

class MoldRfInfoRepository {
    suspend fun getTagInfo(tag: String): TagInfo? {
        return try {
            ApiClient.get("foampad/moldpda/taginfo?MOLD_CODE=$tag")
        } catch (e: Exception) {
            null
        }
    }
}

data class TagInfo(
    val mold_code: String,
    val mold_name: String,
    val car_code: String,
    val code_value1: String
)


