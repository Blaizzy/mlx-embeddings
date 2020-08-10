# frozen_string_literal: true

require "utils/github"

module SPDX
  module_function

  JSON_PATH = (HOMEBREW_LIBRARY_PATH/"data/spdx.json").freeze
  API_URL = "https://api.github.com/repos/spdx/license-list-data/releases/latest"

  def spdx_data
    @spdx_data ||= JSON.parse(JSON_PATH.read)
  end

  def download_latest_license_data!(to: JSON_PATH)
    latest_tag = GitHub.open_api(API_URL)["tag_name"]
    data_url = "https://raw.githubusercontent.com/spdx/license-list-data/#{latest_tag}/json/licenses.json"
    curl_download(data_url, to: to, partial: false)
  end
end
