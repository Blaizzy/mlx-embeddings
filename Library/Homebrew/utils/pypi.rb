# typed: false
# frozen_string_literal: true

# Helper functions for updating PyPI resources.
#
# @api private
module PyPI
  module_function

  PYTHONHOSTED_URL_PREFIX = "https://files.pythonhosted.org/packages/"
  private_constant :PYTHONHOSTED_URL_PREFIX

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

  # Get name, URL, SHA-256 checksum, and latest version for a given PyPI package.
  def get_pypi_info(package, version = nil)
    package = package.split("[").first
    metadata_url = if version.present?
      "https://pypi.org/pypi/#{package}/#{version}/json"
    else
      "https://pypi.org/pypi/#{package}/json"
    end
    out, _, status = curl_output metadata_url, "--location"

    return unless status.success?

    begin
      json = JSON.parse out
    rescue JSON::ParserError
      return
    end

    sdist = json["urls"].find { |url| url["packagetype"] == "sdist" }
    return json["info"]["name"] if sdist.nil?

    [json["info"]["name"], sdist["url"], sdist["digests"]["sha256"], json["info"]["version"]]
  end

  # Return true if resources were checked (even if no change).
  def update_python_resources!(formula, version: nil, package_name: nil, extra_packages: nil, exclude_packages: nil,
                               print_only: false, silent: false, ignore_non_pypi_packages: false)

    auto_update_list = formula.tap.formula_lists[:pypi_automatic_resource_update_list]
    if auto_update_list.present? && auto_update_list.key?(formula.full_name) &&
       package_name.blank? && extra_packages.blank? && exclude_packages.blank?

      list_entry = auto_update_list[formula.full_name]
      case list_entry
      when false
        unless print_only
          odie "The resources for \"#{formula.name}\" need special attention. Please update them manually."
        end
      when String
        package_name = list_entry
      when Hash
        package_name = list_entry["package_name"]
        extra_packages = list_entry["extra_packages"]
        exclude_packages = list_entry["exclude_packages"]
      end
    end

    version ||= formula.version if package_name.blank?
    package_name ||= url_to_pypi_package_name formula.stable.url
    extra_packages ||= []
    exclude_packages ||= []

    if package_name.blank?
      return if ignore_non_pypi_packages

      odie <<~EOS
        Could not infer PyPI package name from URL:
          #{Formatter.url(formula.stable.url)}
      EOS
    end

    input_package_names = { package_name => version }
    extra_packages.each do |extra|
      extra_name, extra_version = extra.split "=="

      if input_package_names.key?(extra_name) && input_package_names[extra_name] != extra_version
        odie "Conflicting versions specified for the `#{extra_name}` package: "\
              "#{input_package_names[extra_name]}, #{extra_version}"
      end

      input_package_names[extra_name] = extra_version
    end

    input_package_names.each do |name, package_version|
      name = name.split("[").first
      next if get_pypi_info(name, package_version).present?

      version_string = " at version #{package_version}" if package_version.present?
      odie "\"#{name}\"#{version_string} is not available on PyPI." unless ignore_non_pypi_packages
    end

    non_pypi_resources = formula.resources.reject do |resource|
      resource.url.start_with? PYTHONHOSTED_URL_PREFIX
    end

    if non_pypi_resources.present? && !print_only
      odie "\"#{formula.name}\" contains non-PyPI resources. Please update the resources manually."
    end

    @pipgrip_installed ||= Formula["pipgrip"].any_version_installed?
    odie '"pipgrip" must be installed (`brew install pipgrip`)' unless @pipgrip_installed

    found_packages = {}
    input_package_names.each do |name, package_version|
      pypi_package_string = if package_version.present?
        "#{name}==#{package_version}"
      else
        name
      end

      ohai "Retrieving PyPI dependencies for \"#{pypi_package_string}\"..." if !print_only && !silent
      pipgrip_output = Utils.popen_read Formula["pipgrip"].bin/"pipgrip", "--json", "--no-cache-dir",
                                        pypi_package_string
      unless $CHILD_STATUS.success?
        odie <<~EOS
          Unable to determine dependencies for \"#{name}\" because of a failure when running
          `pipgrip --json --no-cache-dir #{pypi_package_string}`.
          Please update the resources for \"#{formula.name}\" manually.
        EOS
      end

      found_packages.merge!(JSON.parse(pipgrip_output).to_h) do |conflicting_package, old_version, new_version|
        next old_version if old_version == new_version

        odie "Conflicting versions found for the `#{conflicting_package}` resource: #{old_version}, #{new_version}"
      end
    end

    # Remove extra packages that may be included in pipgrip output
    exclude_list = %W[#{package_name.split("[").first.downcase} argparse pip setuptools wheel wsgiref]
    found_packages.delete_if { |package| exclude_list.include? package }

    new_resource_blocks = ""
    found_packages.sort.each do |package, package_version|
      if exclude_packages.include? package
        ohai "Excluding \"#{package}==#{package_version}\"" if !print_only && !silent
        next
      end

      ohai "Getting PyPI info for \"#{package}==#{package_version}\"" if !print_only && !silent
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

    # Check whether resources already exist (excluding virtualenv dependencies)
    if formula.resources.all? { |resource| resource.name.start_with?("homebrew-") }
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

    true
  end
end
