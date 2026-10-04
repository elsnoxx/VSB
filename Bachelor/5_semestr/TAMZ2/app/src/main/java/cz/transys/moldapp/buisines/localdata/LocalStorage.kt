package cz.transys.moldapp.buisines.localdata

import android.content.Context

class LocalStorage(context: Context) {
    private val prefs = context.getSharedPreferences("moldapp_prefs", Context.MODE_PRIVATE)

    fun saveUserId(userId: String) {
        prefs.edit().putString("user_id", userId).apply()
    }

    fun getUserId(): String? {
        return prefs.getString("user_id", null)
    }

    fun clearUserId() {
        prefs.edit().remove("user_id").apply()
    }

    fun saveUserName(userId: String) {
        prefs.edit().putString("user_name", userId).apply()
    }

    fun getUserName(): String? {
        return prefs.getString("user_name", null)
    }

    fun clearUserName() {
        prefs.edit().remove("user_name").apply()
    }

    fun saveJwtToken(token: String){
        prefs.edit().putString("token", token).apply()
    }

    fun getJwtToken(): String? {
        return prefs.getString("token", null)
    }

    fun clearJwtToken() {
        prefs.edit().remove("user_id").apply()
    }
}