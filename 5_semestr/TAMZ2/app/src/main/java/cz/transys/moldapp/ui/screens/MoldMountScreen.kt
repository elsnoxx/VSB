package cz.transys.moldapp.ui.screens

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import cz.transys.moldapp.LocalScanner
import cz.transys.moldapp.R
import cz.transys.moldapp.buisines.apicalls.moldapi.CarrierMountResponse
import cz.transys.moldapp.buisines.apicalls.moldapi.CarriersList
import cz.transys.moldapp.buisines.apicalls.moldapi.MoldApiRepository
import cz.transys.moldapp.buisines.apicalls.rfidinfo.MoldRfInfoRepository
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MoldMountScreen(onBack: () -> Unit) {
    val colors = MaterialTheme.colorScheme
    val scope = rememberCoroutineScope()
    val repo = remember { MoldApiRepository() }
    val rfRepo = remember { MoldRfInfoRepository() }

    var showDuplicateDialog by remember { mutableStateOf(false) }

    val scanner = LocalScanner.current

    // Carrier dropdown
    var expandedCarrier by remember { mutableStateOf(false) }
    var selectedCarrier by remember { mutableStateOf("") }

    var carrierList by remember { mutableStateOf<List<CarriersList>>(emptyList()) }
    var mountInfo by remember { mutableStateOf<CarrierMountResponse?>(null) }

    // MOLD #1 (read-only)
    var mold1Type by remember { mutableStateOf("") }
    var mold1Code by remember { mutableStateOf("") }
    var carCode1 by remember { mutableStateOf("") }

    // MOLD #2 (editable – new mount)
    var mold2Type by remember { mutableStateOf("") }
    var mold2Code by remember { mutableStateOf("") }
    var carCode2 by remember { mutableStateOf("") }

    LaunchedEffect(scanner, selectedCarrier) {
        scanner?.setOnScanListener { scannedTag ->

            val tag = scannedTag.trim()

            // NESMÍ být prázdný
            if (tag.isBlank()) return@setOnScanListener

            if (tag == mold1Code || tag == mold2Code) {

                showDuplicateDialog = true

                scope.launch {
                    showDuplicateDialog = false
                }

                return@setOnScanListener
            }


            // LOAD TAG INFO FROM API
            scope.launch {
                val info = rfRepo.getTagInfo(tag)

                if (info != null) {

                    // FIRST SLOT EMPTY? → FILL MOLD #1
                    if (mold1Code.isBlank()) {
                        mold1Code = info.mold_code
                        mold1Type = info.mold_name
                        carCode1 = info.car_code
                    }

                    // SECOND SLOT EMPTY? → FILL MOLD #2
                    else if (mold2Code.isBlank()) {
                        mold2Code = info.mold_code
                        mold2Type = info.mold_name
                        carCode2 = info.car_code
                    }

                } else {
                    Log.e("RFID", "Tag not found in API")
                }
            }
        }
    }

    DisposableEffect(scanner) {
        onDispose { scanner?.setOnScanListener { } }
    }


    // Load carriers
    LaunchedEffect(Unit) {
        try {
            carrierList = repo.getAllCarriers()
        } catch (_: Exception) {}
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(colors.background)
            .padding(16.dp),
        verticalArrangement = Arrangement.Top,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        if (showDuplicateDialog) {
            AlertDialog(
                onDismissRequest = { showDuplicateDialog = false },
                title = { Text(stringResource(R.string.dialog_mold_mount_title)) },
                text = { Text(stringResource(R.string.dialog_mold_mount_text)) },
                confirmButton = {
                    TextButton(onClick = { showDuplicateDialog = false }) {
                        Text(stringResource(R.string.dialog_mold_mount_button))
                    }
                }
            )
        }

        Text(
            text = stringResource(R.string.mold_mount_title),
            fontSize = 26.sp,
            fontWeight = FontWeight.Bold,
            color = colors.primary,
            modifier = Modifier.padding(bottom = 20.dp)
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(colors.surface)
                .padding(16.dp)
        ) {

            // CARRIER SELECT
            ExposedDropdownMenuBox(
                expanded = expandedCarrier,
                onExpandedChange = { expandedCarrier = !expandedCarrier }
            ) {
                OutlinedTextField(
                    value = selectedCarrier,
                    onValueChange = {},
                    readOnly = true,
                    label = { Text(stringResource(R.string.carrier_label)) },
                    trailingIcon = {
                        ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedCarrier)
                    },
                    modifier = Modifier
                        .menuAnchor()
                        .fillMaxWidth()
                )

                ExposedDropdownMenu(
                    expanded = expandedCarrier,
                    onDismissRequest = { expandedCarrier = false }
                ) {
                    if (carrierList.isEmpty()) {
                        DropdownMenuItem(
                            text = { Text("No carriers loaded") },
                            onClick = {}
                        )
                    } else {
                        carrierList.forEach { carrier ->
                            DropdownMenuItem(
                                text = { Text(carrier.code_value2) },
                                onClick = {
                                    selectedCarrier = carrier.code_value1
                                    expandedCarrier = false
                                }
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // MOLD #1 – read only
            MoldPartChangeSection(
                title = stringResource(R.string.mold1_title),
                color = colors.tertiary,
                carCode = carCode1,
                type = mold1Type,
                code = mold1Code,
                readOnlyCode = true,
                onCodeChange = {}
            )

            Spacer(modifier = Modifier.height(12.dp))

            // MOLD #2 – editable (new mold to mount)
            MoldPartChangeSection(
                title = stringResource(R.string.mold2_title),
                color = colors.error,
                carCode = carCode2,
                type = mold2Type,
                code = mold2Code,
                readOnlyCode = false,
                onCodeChange = { mold2Code = it }
            )

            Spacer(modifier = Modifier.height(20.dp))

            Button(
                onClick = { /* TODO API mount*/ },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(60.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = colors.primary,
                    contentColor = colors.onPrimary
                )
            ) {
                Text(
                    text = stringResource(R.string.mount_button),
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = onBack,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = colors.secondary,
                    contentColor = colors.onSecondary
                )
            ) {
                Text(stringResource(R.string.back_button))
            }
        }
    }
}