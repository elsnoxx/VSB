package cz.transys.moldapp.ui.screens

import android.content.Context
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavHostController
import cz.transys.moldapp.R

@Composable
fun MenuScreen(context: Context, navController: NavHostController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(8.dp),
        verticalArrangement = Arrangement.SpaceEvenly
    ) {
        Row(modifier = Modifier.weight(1f)) {
            CustomButton(R.string.menu_tag_write, R.drawable.writeblack, Modifier.weight(1f)) {
                navController.navigate("tag_write")
            }
            CustomButton(R.string.menu_mold_repair, R.drawable.repairblack, Modifier.weight(1f)) {
                navController.navigate("mold_repair")
            }
        }

        Row(modifier = Modifier.weight(1f)) {
            CustomButton(R.string.menu_part_change, R.drawable.changeblack, Modifier.weight(1f)) {
                navController.navigate("part_change")
            }
            CustomButton(R.string.menu_mold_mount, R.drawable.mountblack, Modifier.weight(1f)) {
                navController.navigate("mold_mount")
            }
        }

        Row(modifier = Modifier.weight(1f)) {
            CustomButton(R.string.menu_rf_tag_info, R.drawable.infoblack, Modifier.weight(1f)) {
                navController.navigate("rf_tag_info")
            }
            CustomButton(R.string.menu_test_reading, R.drawable.rfid, Modifier.weight(1f)) {
                navController.navigate("test_reading")
            }
        }
    }
}

@Composable
fun CustomButton(
    nameResId: Int,
    imageRes: Int? = null,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    val name = stringResource(id = nameResId)
    val colors = MaterialTheme.colorScheme

    Button(
        onClick = onClick,
        shape = RectangleShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = colors.primary,
            contentColor = colors.onPrimary
        ),
        modifier = modifier
            .padding(4.dp)
            .fillMaxSize()
    ) {
        Column(
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            if (imageRes != null) {
                Image(
                    painter = painterResource(id = imageRes),
                    contentDescription = name,
                    modifier = Modifier.size(70.dp)
                )
            } else {
                Box(
                    modifier = Modifier
                        .size(70.dp)
                        .background(Color.Gray)
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = name,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = colors.onPrimary
            )
        }
    }
}
