<?php

//header("refresh: 2");

define("MYSQL_SERVER", "localhost");
define("MYSQL_USER", "id20856516_eurodollarbot");
define("MYSQL_PASSWORD", "replaceme");
define("MYSQL_DB", "id20856516_eurodollarbot");
define("TBL_PREFIX", "eurodollarbot");
define("CREATE_DB_IF_NOT_EXISTS", true);
define("CREATE_TABLES_IF_NOT_EXIST", true);
define("LOG_IP", true);
define("LOG_IP_IGNORE", "127.0.0.1");
define("DISABLE_DETAILED_LOG_VIEW", true);
define("DEBUG", true);
define("SHOW_ONLY_TODAY", true);

//echo basename($_SERVER['PHP_SELF']);

if (LOG_IP==true){
    $db = new mysqli(MYSQL_SERVER, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB);
    if ($db->connect_errno) {
        exit;
    }
    $client_ip = $_SERVER['REMOTE_ADDR'];
    $nslookup = gethostbyaddr($client_ip);
    $url = $_SERVER['PHP_SELF'];
    $db->close();
}

$arrOptions=array(
    "ssl"=>array(
        "verify_peer"=>false,
        "verify_peer_name"=>false,
    ),
);

// $page = file_get_contents("https://rates.fxcm.com/RatesXML", false, stream_context_create($arrOptions));
// $xml = new SimpleXMLElement($page);
// $result = $xml->xpath('/Rates/Rate');
// echo 'result count = ' . count($result);
// echo '<br/>' . 'Live FXCM rates' . '<br/>';                 
// //$rates=array();
// for($i=0;$i<count($result);$i++){
//     $symbol = (string) $result[$i]->xpath('@Symbol')[0];
//     $bid = (string) $result[$i]->xpath('Bid')[0]; // sell price
//     $ask = (string) $result[$i]->xpath('Ask')[0]; // buy price
//     $rates[$i]["symbol"]=$symbol;
//     $rates[$i]["bid"]=$bid;
//     $rates[$i]["ask"]=$ask;

//     //echo $rates[$i][$symbol];
//     echo $symbol . ' ' . $bid . '<br/>';
// }
// $title = "";

$db = new mysqli(MYSQL_SERVER, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB);
if ($db->connect_errno) {
    echo "Erreur : " . $db->connect_errno . " <br/>";
    exit;
}


if (isset($_GET['create_db'])) {
    echo "create db";
    $db = new mysqli(MYSQL_SERVER, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB);
    if ($db->connect_errno) {
        exit;
    }
    $sql = "CREATE TABLE " . TBL_PREFIX . "_notification (`timestamp` text COLLATE utf8_unicode_ci NOT NULL, `message` text COLLATE utf8_unicode_ci NOT NULL ) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;";
    $r = mysqli_query($db, $sql);
    // $sql = "CREATE TABLE " . TBL_PREFIX . "_ssb_alert (`timestamp` text COLLATE utf8_unicode_ci NOT NULL, `period` text COLLATE utf8_unicode_ci NOT NULL, `name` text COLLATE utf8_unicode_ci NOT NULL, `type` text COLLATE utf8_unicode_ci NOT NULL, `price` double NOT NULL, `ssb` double NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;";
    // $r = mysqli_query($db, $sql);
    // $sql = "CREATE TABLE " . TBL_PREFIX . "_ip_address_log (`id` bigint(20) NOT NULL AUTO_INCREMENT, `access_date_time` datetime NOT NULL, `ip_address` varchar(32) COLLATE latin1_general_ci NOT NULL, `nslookup` text, `url` varchar(255) COLLATE latin1_general_ci DEFAULT NULL, `count` bigint(20), PRIMARY KEY (`id`)) ENGINE=MyISAM  DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci";
    // $r = mysqli_query($db, $sql);
    // $sql = "CREATE TABLE " . TBL_PREFIX . "_2jcs_alert (`id` bigint(20) NOT NULL, `timestamp` text COLLATE utf8_unicode_ci NOT NULL, `period` text COLLATE utf8_unicode_ci NOT NULL, `symbol` text COLLATE utf8_unicode_ci NOT NULL, `buy` double NOT NULL,`sell` double NOT NULL, `h1_ls_validated` text COLLATE utf8_unicode_ci NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;";
    // $r = mysqli_query($db, $sql);
    // $sql = "ALTER TABLE " . TBL_PREFIX . "_2jcs_alert ADD PRIMARY KEY (`id`);";
    // $r = mysqli_query($db, $sql);
    // $sql = "ALTER TABLE " . TBL_PREFIX . "_2jcs_alert MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;";
    // $r = mysqli_query($db, $sql);
    // $sql= "ALTER TABLE " . TBL_PREFIX . "_2jcs_alert ADD `m1_ls_validated` TEXT NOT NULL  AFTER `h1_ls_validated`";
    // $r = mysqli_query($db, $sql);
    // $sql="CREATE TABLE " . TBL_PREFIX . "_history (`id` bigint(20) NOT NULL,`timestamp` text COLLATE utf8_unicode_ci NOT NULL,`symbol` text COLLATE utf8_unicode_ci NOT NULL,`buy` double NOT NULL,`sell` double NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;";
    // $r = mysqli_query($db, $sql);
    // $sql="ALTER TABLE " . TBL_PREFIX . "_history ADD PRIMARY KEY (`id`);";
    // $r = mysqli_query($db, $sql);
    // $sql="ALTER TABLE " . TBL_PREFIX . "_history MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT;";
    // $r = mysqli_query($db, $sql);
    $db->close();
}

if (isset($_GET['upload_history'])) {
    $history = $_GET['upload_history'];
    echo "received=  [[$history]]<br/>";
    $array = explode(";", $history);
    if (count($array)==2){
        $timestamp = $array[0];
        $message = $array[1];
        $db = new mysqli(MYSQL_SERVER, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB);
        if ($db->connect_errno) {
            echo "Error : " . $db->connect_errno . " <br/>";
            exit;
        }
        //$timestamp = (new DateTime())->format('Y-m-d H:i:s');
        $r = mysqli_query($db, "insert into " . TBL_PREFIX . "_notification(timestamp, message) values ('" . $timestamp . "', '" . $message . "')");
        echo 'History recorded OK.<br/>';
        $db->close();
        exit;
    }
}

?>
