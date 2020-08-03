# frozen_string_literal: true

module PyPI
  module_function

  PYTHONHOSTED_URL_PREFIX = "https://files.pythonhosted.org/packages/"

  AUTOMATIC_RESOURCE_UPDATE_BLOCKLIST = %w[
    ansible
    ansible@2.8
    cloudformation-cli
    diffoscope
    dxpy
    molecule
  ].freeze

  @pipgrip_installed = nil

  def url_to_pypi_package_name(url)
    return unless url.start_with? PYTHONHOSTED_URL_PREFIX

    File.basename(url).match(/^(.+)-[a-z\d.]+$/)[1]
  end

  def update_pypi_url(url, version)
    package = url_to_pypi_package_name url
    return if package.nil?

    _, url = get_pypi_info(package, version)
    url
  end

  # Get name, url and sha256 for a given pypi package
  def get_pypi_info(package, version)
    metadata_url = "https://pypi.org/pypi/#{package}/#{version}/json"
    out, _, status = curl_output metadata_url, "--location"

    return unless status.success?

    begin
      json = JSON.parse out
    rescue JSON::ParserError
      return
    end

    sdist = json["urls"].find { |url| url["packagetype"] == "sdist" }
    return json["info"]["name"] if sdist.nil?

    [json["info"]["name"], sdist["url"], sdist["digests"]["sha256"]]
  end

  def update_python_resources!(formula, version = nil, print_only: false, silent: false,
                               ignore_non_pypi_packages: false)

    if !print_only && AUTOMATIC_RESOURCE_UPDATE_BLOCKLIST.include?(formula.full_name)
      odie "The resources for \"#{formula.name}\" need special attention. Please update them manually."
      return
    end

    # PyPI package name isn't always the same as the formula name. Try to infer from the URL.
    pypi_name = if formula.stable.url.start_with?(PYTHONHOSTED_URL_PREFIX)
      url_to_pypi_package_name formula.stable.url
    else
      formula.name
    end

    version ||= formula.version

    if get_pypi_info(pypi_name, version).blank?
      odie "\"#{pypi_name}\" at version #{version} is not available on PyPI." unless ignore_non_pypi_packages
      return
    end

    non_pypi_resources = formula.resources.reject do |resource|
      resource.url.start_with? PYTHONHOSTED_URL_PREFIX
    end

    if non_pypi_resources.present? && !print_only
      odie "\"#{formula.name}\" contains non-PyPI resources. Please update the resources manually."
    end

    @pipgrip_installed ||= Formula["pipgrip"].any_version_installed?
    odie '"pipgrip" must be installed (`brew install pipgrip`)' unless @pipgrip_installed

    ohai "Retrieving PyPI dependencies for \"#{pypi_name}==#{version}\"" if !print_only && !silent
    pipgrip_output = Utils.popen_read Formula["pipgrip"].bin/"pipgrip", "--json", "--no-cache-dir",
                                      "#{pypi_name}==#{version}"
    unless $CHILD_STATUS.success?
      odie <<~EOS
        Unable to determine dependencies for \"#{pypi_name}\" because of a failure when running
        `pipgrip --json --no-cache-dir #{pypi_name}==#{version}`.
        Please update the resources for \"#{formula.name}\" manually.
      EOS
    end

    packages = JSON.parse(pipgrip_output).sort.to_h

    # Remove extra packages that may be included in pipgrip output
    exclude_list = %W[#{pypi_name} argparse pip setuptools wheel wsgiref]
    packages.delete_if do |package|
      exclude_list.include? package
    end

    new_resource_blocks = ""
    packages.each do |package, package_version|
      name, url, checksum = get_pypi_info package, package_version
      # Fail if unable to find name, url or checksum for any resource
      if name.blank?
        odie "Unable to resolve some dependencies. Please update the resources for \"#{formula.name}\" manually."
      elsif url.blank? || checksum.blank?
        odie <<~EOS
          Unable to find the URL and/or sha256 for the \"#{name}\" resource.
          Please update the resources for \"#{formula.name}\" manually.
        EOS
      end

      # Append indented resource block
      new_resource_blocks += <<-EOS
  resource "#{name}" do
    url "#{url}"
    sha256 "#{checksum}"
  end

      EOS
    end

    if print_only
      puts new_resource_blocks.chomp
      return
    end

    # Check whether resources already exist (excluding homebrew-virtualenv)
    if formula.resources.blank? ||
       (formula.resources.length == 1 && formula.resources.first.name == "homebrew-virtualenv")
      # Place resources above install method
      inreplace_regex = /  def install/
      new_resource_blocks += "  def install"
    else
      # Replace existing resource blocks with new resource blocks
      inreplace_regex = /  (resource .* do\s+url .*\s+sha256 .*\s+ end\s*)+/
      new_resource_blocks += "  "
    end

    ohai "Updating resource blocks" unless silent
    Utils::Inreplace.inreplace formula.path do |s|
      if s.inreplace_string.scan(inreplace_regex).length > 1
        odie "Unable to update resource blocks for \"#{formula.name}\" automatically. Please update them manually."
      end
      s.sub! inreplace_regex, new_resource_blocks
    end
  end
end
