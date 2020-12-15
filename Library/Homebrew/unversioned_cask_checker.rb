# typed: true
# frozen_string_literal: true

require "bundle_version"
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
          Pathname.glob(dir/"**"/app.source.basename/"Contents"/"Info.plist").reject do |info_plist_path|
            # Ignore nested apps.
            info_plist_path.parent.parent.parent.ascend.any? { |p| p.extname == ".app" }
          end.sort
        end

        info_plist_paths.each do |info_plist_path|
          if (version = BundleVersion.from_info_plist(info_plist_path)&.nice_version)
            return version
          end
        end

        pkg_paths = pkgs.flat_map do |pkg|
          Pathname.glob(dir/"**"/pkg.path.basename).sort
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
              if (version = BundleVersion.from_package_info(package_info_path)&.nice_version)
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
