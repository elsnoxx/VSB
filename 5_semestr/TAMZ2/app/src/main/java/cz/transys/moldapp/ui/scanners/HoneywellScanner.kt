package cz.transys.moldapp.ui.scanners

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.MediaPlayer
import android.os.Bundle
import android.util.Log
import androidx.core.content.ContextCompat
import kotlin.math.log


//class HoneywellScanner(
//    private val context: Context,
//    private val scanListener: RawScanHandler
//): Scanner {

class HoneywellScanner(
    private val context: Context,
) {

    private var failTonePlayer: MediaPlayer? = null
    private var onScanCallback: ((String) -> Unit)? = null

    companion object {
        private const val ACTION_BARCODE_DATA: String = "com.honeywell.sample.action.BARCODE_DATA"
        private const val ACTION_CLAIM_SCANNER: String =
            "com.honeywell.aidc.action.ACTION_CLAIM_SCANNER"
        private const val ACTION_RELEASE_SCANNER: String =
            "com.honeywell.aidc.action.ACTION_RELEASE_SCANNER"
        private const val EXTRA_SCANNER: String = "com.honeywell.aidc.extra.EXTRA_SCANNER"
        private const val EXTRA_PROFILE: String = "com.honeywell.aidc.extra.EXTRA_PROFILE"
        private const val EXTRA_PROPERTIES: String = "com.honeywell.aidc.extra.EXTRA_PROPERTIES"
    }

    fun setOnScanListener(listener: (String) -> Unit) {
        onScanCallback = listener
    }

    private val barcodeDataReceiver: BroadcastReceiver = object: BroadcastReceiver(){
        override fun onReceive(context: Context, intent: Intent) {
            if(ACTION_BARCODE_DATA == intent.action){
                handleScan(intent)
            }
        }
    }

    private fun handleScan(intent: Intent) {
        val data = intent.getStringExtra("data")
        if(data.isNullOrEmpty()) {
//            handleError("Incorrect QR", "Error parsing QR Code!")
            Log.d("Incorrect QR", "Error parsing QR Code!")
        } else {
//            scanListener.onScan(this, data)
            Log.d("scanned",data )
            onScanCallback?.invoke(data)
        }
    }

    fun open() {
        ContextCompat.registerReceiver(
            context,
            barcodeDataReceiver,
            IntentFilter(ACTION_BARCODE_DATA),
            ContextCompat.RECEIVER_NOT_EXPORTED
        )
        claimScanner()
//        failTonePlayer = MediaPlayer.create(context, R.raw.fail2)
    }

    fun close() {
        context.unregisterReceiver(barcodeDataReceiver)
        releaseScanner()
        failTonePlayer?.release()
        failTonePlayer = null
    }

    private fun claimScanner() {
        val properties = Bundle()
        properties.putBoolean("DPR_DATA_INTENT", true)
        properties.putString("DPR_DATA_INTENT_ACTION", ACTION_BARCODE_DATA)
        context.sendBroadcast(Intent(ACTION_CLAIM_SCANNER)
            .putExtra(EXTRA_SCANNER, "dcs.scanner.imager")
            .putExtra(EXTRA_PROFILE, "myprofile1")
            .putExtra(EXTRA_PROPERTIES, properties))
    }

    private fun releaseScanner(){
        context.sendBroadcast(Intent(ACTION_RELEASE_SCANNER))
    }

//    override fun handleSuccess() {
//
//    }

//    override fun handleError(shortMessage: String, longMessage: String) {
//        ApiDialogs.showErrorDialog(context, longMessage, shortMessage)
//        failTonePlayer?.start()
//    }
}
