# typed: strict

module SPDX
  include Kernel

  def spdx_data; end

  def download_latest_license_data!(to: JSON_PATH); end

  def curl_download(*args, to: nil, partial: true, **options); end
end
