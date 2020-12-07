# typed: true
# frozen_string_literal: true

require "cask/cask"
require "cask/installer"

module Homebrew
  # Check unversioned casks for updates by extracting their
  # contents and guessing the version from contained files.
  #
  # @api private
  class UnversionedCaskChecker
    extend T::Sig

    sig {  returns(Cask::Cask) }
    attr_reader :cask

    sig { params(cask: Cask::Cask).void }
    def initialize(cask)
      @cask = cask
    end

    sig { returns(Cask::Installer) }
    def installer
      @installer ||= Cask::Installer.new(cask, verify_download_integrity: false)
    end

    sig { returns(T::Array[Cask::Artifact::App]) }
    def apps
      @apps ||= @cask.artifacts.select { |a| a.is_a?(Cask::Artifact::App) }
    end

    sig { returns(T::Array[Cask::Artifact::Pkg]) }
    def pkgs
      @pkgs ||= @cask.artifacts.select { |a| a.is_a?(Cask::Artifact::Pkg) }
    end

    sig { returns(T::Boolean) }
    def single_app_cask?
      apps.count == 1
    end

    sig { returns(T::Boolean) }
    def single_pkg_cask?
      pkgs.count == 1
    end

    sig { params(info_plist_path: Pathname).returns(T.nilable(String)) }
    def self.version_from_info_plist(info_plist_path)
      plist = system_command!("plutil", args: ["-convert", "xml1", "-o", "-", info_plist_path]).plist

      short_version = plist["CFBundleShortVersionString"].presence
      version = plist["CFBundleVersion"].presence

      return decide_between_versions(short_version, version) if short_version && version
    end

    sig { params(package_info_path: Pathname).returns(T.nilable(String)) }
    def self.version_from_package_info(package_info_path)
      contents = package_info_path.read

      short_version = contents[/CFBundleShortVersionString="([^"]*)"/, 1].presence
      version = contents[/CFBundleVersion="([^"]*)"/, 1].presence

      return decide_between_versions(short_version, version) if short_version && version
    end

    sig do
      params(short_version: T.nilable(String), version: T.nilable(String))
        .returns(T.nilable(String))
    end
    def self.decide_between_versions(short_version, version)
      return short_version if short_version == version

      short_version_match = short_version&.match?(/\A\d+(\.\d+)+\Z/)
      version_match = version&.match?(/\A\d+(\.\d+)+\Z/)

      if short_version_match && version_match
        return version if version.length > short_version.length && version.start_with?("#{short_version}.")
        return short_version if short_version.length > version.length && short_version.start_with?("#{version}.")
      end

      if short_version&.match?(/\A\d+(\.\d+)*\Z/) && version&.match?(/\A\d+\Z/)
        return short_version if short_version.start_with?("#{version}.") || short_version.end_with?(".#{version}")

        return "#{short_version},#{version}"
      end

      short_version || version
    end

    sig { returns(T.nilable(String)) }
    def guess_cask_version
      if apps.empty? && pkgs.empty?
        opoo "Cask #{cask} does not contain any apps or PKG installers."
        return
      end

      Dir.mktmpdir do |dir|
        dir = Pathname(dir)

        installer.yield_self do |i|
          i.extract_primary_container(to: dir)
        rescue ErrorDuringExecution => e
          onoe e
          return nil
        end

        info_plist_paths = apps.flat_map do |app|
          Pathname.glob(dir/"**"/app.source.basename/"Contents"/"Info.plist")
        end

        info_plist_paths.each do |info_plist_path|
          if (version = self.class.version_from_info_plist(info_plist_path))
            return version
          end
        end

        pkg_paths = pkgs.flat_map do |pkg|
          Pathname.glob(dir/"**"/pkg.path.basename)
        end

        pkg_paths.each do |pkg_path|
          packages =
            system_command!("installer", args: ["-plist", "-pkginfo", "-pkg", pkg_path])
            .plist
            .map { |package| package.fetch("Package") }
            .uniq

          Dir.mktmpdir do |extract_dir|
            extract_dir = Pathname(extract_dir)
            FileUtils.rmdir extract_dir

            begin
              system_command! "pkgutil", args: ["--expand-full", pkg_path, extract_dir]
            rescue ErrorDuringExecution => e
              onoe "Failed to extract #{pkg_path.basename}: #{e}"
              next
            end

            package_info_path = extract_dir/"PackageInfo"
            if package_info_path.exist?
              if (version = self.class.version_from_package_info(package_info_path))
                return version
              end
            elsif packages.count == 1
              onoe "#{pkg_path.basename} does not contain a `PackageInfo` file."
            end

            opoo "#{pkg_path.basename} contains multiple packages: (#{packages.join(", ")})" if packages.count != 1

            $stderr.puts Pathname.glob(extract_dir/"**/*")
                                 .map { |path|
                                   regex = %r{\A(.*?\.(app|qlgenerator|saver|plugin|kext|bundle|osax))/.*\Z}
                                   path.to_s.sub(regex, '\1')
                                 }.uniq
          ensure
            Cask::Utils.gain_permissions_remove(extract_dir)
            extract_dir.mkpath
          end
        end

        nil
      end
    end
  end
end
