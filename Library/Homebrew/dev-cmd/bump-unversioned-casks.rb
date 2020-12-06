# typed: false
# frozen_string_literal: true

require "cask/download"
require "cask/installer"
require "cask/cask_loader"
require "cli/parser"
require "tap"

module Homebrew
  extend T::Sig

  extend SystemCommand::Mixin

  sig { returns(CLI::Parser) }
  def self.bump_unversioned_casks_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump-unversioned-casks` [<options>] [<tap>]

        Check all casks with unversioned URLs in a given <tap> for updates.
      EOS
      switch "-n", "--dry-run",
             description: "List what would be done, but do not actually do anything."
      flag  "--limit=",
            description: "Maximum runtime in minutes."
      flag   "--state-file=",
             description: "File for keeping track of state."

      named 1
    end
  end

  sig { void }
  def self.bump_unversioned_casks
    args = bump_unversioned_casks_args.parse

    state_file = if args.state_file.present?
      Pathname(args.state_file).expand_path
    else
      HOMEBREW_CACHE/"bump_unversioned_casks.json"
    end
    state_file.dirname.mkpath

    tap = Tap.fetch(args.named.first)

    state = state_file.exist? ? JSON.parse(state_file.read) : {}

    cask_files = tap.cask_files
    unversioned_cask_files = cask_files.select do |cask_file|
      url = cask_file.each_line do |line|
        url = line[/\s*url\s+"([^"]+)"\s*/, 1]
        break url if url
      end

      url.present? && url.exclude?('#{')
    end.sort

    unversioned_casks = unversioned_cask_files.map { |path| Cask::CaskLoader.load(path) }

    ohai "Unversioned Casks:"
    puts "Total:      #{unversioned_casks.count}"
    puts "Single-App: #{unversioned_casks.count { |c| single_app_cask?(c) }}"
    puts "Single-Pkg: #{unversioned_casks.count { |c| single_pkg_cask?(c) }}"

    checked, unchecked = unversioned_casks.partition { |c| state.key?(c.full_name) }

    queue = Queue.new

    unchecked.shuffle.each do |c|
      queue.enq c
    end
    checked.sort_by { |c| state.dig(c.full_name, "check_time") }.each do |c|
      queue.enq c
    end

    limit = args.limit.presence&.to_i
    end_time = Time.now + limit.minutes if limit

    until queue.empty? || (end_time && end_time < Time.now)
      cask = queue.deq

      ohai "Checking #{cask.full_name}"

      unless single_app_cask?(cask) || single_pkg_cask?(cask)
        opoo "Skipping, cask #{cask} it not a single-app or PKG cask."
        next
      end

      last_state = state.fetch(cask.full_name, {})
      last_check_time = last_state["check_time"]&.yield_self { |t| Time.parse(t) }

      check_time = Time.now
      if last_check_time && check_time < (last_check_time + 1.day)
        opoo "Skipping, already checked within the last 24 hours."
        next
      end

      last_sha256 = last_state["sha256"]
      last_time = last_state["time"]&.yield_self { |t| Time.parse(t) }
      last_file_size = last_state["file_size"]

      download = Cask::Download.new(cask)
      time, file_size = begin
        download.time_file_size
      rescue
        [nil, nil]
      end

      if last_time != time || last_file_size != file_size
        installer = Cask::Installer.new(cask, verify_download_integrity: false)

        begin
          cached_download = installer.download
        rescue => e
          onoe e
          next
        end

        sha256 = cached_download.sha256

        if last_sha256 != sha256 && (version = guess_cask_version(cask, installer))
          if cask.version == version
            oh1 "Cask #{cask} is up-to-date at #{version}"
          else
            bump_cask_pr_args = [
              "bump-cask-pr",
              "--version", version.to_s,
              "--sha256", ":no_check",
              "--message", "Automatic update via `brew bump-unversioned-casks`.",
              cask.sourcefile_path
            ]

            if args.dry_run?
              bump_cask_pr_args << "--dry-run"
              oh1 "Would bump #{cask} from #{cask.version} to #{version}"
            else
              oh1 "Bumping #{cask} from #{cask.version} to #{version}"
            end

            begin
              system_command! HOMEBREW_BREW_FILE, args: bump_cask_pr_args
            rescue ErrorDuringExecution => e
              onoe e
              next
            end
          end
        end
      end

      next if args.dry_run?

      state[cask.full_name] = {
        "sha256"     => sha256,
        "check_time" => check_time.iso8601,
        "time"       => time&.iso8601,
        "file_size"  => file_size,
      }

      state_file.atomic_write JSON.generate(state)
    end
  end

  sig { params(cask: Cask::Cask, installer: Cask::Installer).returns(T.nilable(String)) }
  def self.guess_cask_version(cask, installer)
    apps = cask.artifacts.select { |a| a.is_a?(Cask::Artifact::App) }
    pkgs = cask.artifacts.select { |a| a.is_a?(Cask::Artifact::Pkg) }

    if apps.empty? && pkgs.empty?
      opoo "Cask #{cask} does not contain any apps or PKG installers."
      return
    end

    Dir.mktmpdir do |dir|
      dir = Pathname(dir)

      begin
        installer.extract_primary_container(to: dir)
      rescue => e
        onoe e
        next
      end

      info_plist_paths = apps.flat_map do |app|
        Pathname.glob(dir/"**"/app.source.basename/"Contents"/"Info.plist")
      end

      info_plist_paths.each do |info_plist_path|
        if (version = version_from_info_plist(cask, info_plist_path))
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

        if packages.count == 1
          Dir.mktmpdir do |extract_dir|
            extract_dir = Pathname(extract_dir)
            FileUtils.rmdir extract_dir

            system_command! "pkgutil", args: ["--expand-full", pkg_path, extract_dir]

            package_info_path = extract_dir/"PackageInfo"
            if package_info_path.exist?
              if (version = version_from_package_info(cask, package_info_path))
                return version
              end
            else
              onoe "#{pkg_path.basename} does not contain a `PackageInfo` file."
              next
            end
          end
        else
          opoo "Skipping, #{pkg_path.basename} contains multiple packages."
          next
        end
      end

      nil
    end
  end

  sig { params(cask: Cask::Cask, info_plist_path: Pathname).returns(T.nilable(String)) }
  def self.version_from_info_plist(cask, info_plist_path)
    plist = system_command!("plutil", args: ["-convert", "xml1", "-o", "-", info_plist_path]).plist

    short_version = plist["CFBundleShortVersionString"]
    version = plist["CFBundleVersion"]

    return decide_between_versions(cask, short_version, version) if short_version && version
  end

  sig { params(cask: Cask::Cask, package_info_path: Pathname).returns(T.nilable(String)) }
  def self.version_from_package_info(cask, package_info_path)
    contents = package_info_path.read

    short_version = contents[/CFBundleShortVersionString="([^"]*)"/, 1]
    version = contents[/CFBundleVersion="([^"]*)"/, 1]

    return decide_between_versions(cask, short_version, version) if short_version && version
  end

  sig do
    params(cask: Cask::Cask, short_version: T.nilable(String), version: T.nilable(String))
      .returns(T.nilable(String))
  end
  def self.decide_between_versions(cask, short_version, version)
    return "#{short_version},#{version}" if short_version && version && cask.version.include?(",")

    return cask.version.to_s if [short_version, version].include?(cask.version.to_s)

    short_version_match = short_version&.match?(/\A\d+(\.\d+)+\Z/)
    version_match = version&.match?(/\A\d+(\.\d+)+\Z/)

    if short_version_match && version_match
      return version if version.length > short_version.length && version.start_with?(short_version)
      return short_version if short_version.length > version.length && short_version.start_with?(version)
    elsif short_version_match
      return short_version
    elsif version_match
      return version
    end

    short_version || version
  end

  def self.single_app_cask?(cask)
    cask.artifacts.count { |a| a.is_a?(Cask::Artifact::App) } == 1
  end

  def self.single_pkg_cask?(cask)
    cask.artifacts.count { |a| a.is_a?(Cask::Artifact::Pkg) } == 1
  end
end
