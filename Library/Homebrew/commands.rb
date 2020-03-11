# frozen_string_literal: true

module Commands
  module_function

  HOMEBREW_CMD_PATH = (HOMEBREW_LIBRARY_PATH/"cmd").freeze
  HOMEBREW_DEV_CMD_PATH = (HOMEBREW_LIBRARY_PATH/"dev-cmd").freeze
  HOMEBREW_INTERNAL_COMMAND_ALIASES = {
    "ls"          => "list",
    "homepage"    => "home",
    "-S"          => "search",
    "up"          => "update",
    "ln"          => "link",
    "instal"      => "install", # gem does the same
    "uninstal"    => "uninstall",
    "rm"          => "uninstall",
    "remove"      => "uninstall",
    "configure"   => "diy",
    "abv"         => "info",
    "dr"          => "doctor",
    "--repo"      => "--repository",
    "environment" => "--env",
    "--config"    => "config",
    "-v"          => "--version",
  }.freeze

  def valid_internal_cmd?(cmd)
    require?(HOMEBREW_CMD_PATH/cmd)
  end

  def valid_internal_dev_cmd?(cmd)
    require?(HOMEBREW_DEV_CMD_PATH/cmd)
  end

  def method_name(cmd)
    cmd.to_s
       .tr("-", "_")
       .downcase
       .to_sym
  end

  def args_method_name(cmd_path)
    cmd_path_basename = basename_without_extension(cmd_path)
    cmd_method_prefix = method_name(cmd_path_basename)
    "#{cmd_method_prefix}_args".to_sym
  end

  def internal_cmd_path(cmd)
    [
      HOMEBREW_CMD_PATH/"#{cmd}.rb",
      HOMEBREW_CMD_PATH/"#{cmd}.sh",
    ].find(&:exist?)
  end

  def internal_dev_cmd_path(cmd)
    [
      HOMEBREW_DEV_CMD_PATH/"#{cmd}.rb",
      HOMEBREW_DEV_CMD_PATH/"#{cmd}.sh",
    ].find(&:exist?)
  end

  # Ruby commands which can be `require`d without being run.
  def external_ruby_v2_cmd_path(cmd)
    path = which("#{cmd}.rb", Tap.cmd_directories)
    path if require?(path)
  end

  # Ruby commands which are run by being `require`d.
  def external_ruby_cmd_path(cmd)
    which("brew-#{cmd}.rb", PATH.new(ENV["PATH"]).append(Tap.cmd_directories))
  end

  def external_cmd_path(cmd)
    which("brew-#{cmd}", PATH.new(ENV["PATH"]).append(Tap.cmd_directories))
  end

  def path(cmd)
    internal_cmd = HOMEBREW_INTERNAL_COMMAND_ALIASES.fetch(cmd, cmd)
    path ||= internal_cmd_path(internal_cmd)
    path ||= internal_dev_cmd_path(internal_cmd)
    path ||= external_ruby_v2_cmd_path(cmd)
    path ||= external_ruby_cmd_path(cmd)
    path ||= external_cmd_path(cmd)
    path
  end

  def commands(aliases: false)
    cmds = internal_commands
    cmds += internal_developer_commands
    cmds += external_commands
    cmds += internal_commands_aliases if aliases
    cmds.sort
  end

  def internal_commands_paths
    find_commands HOMEBREW_CMD_PATH
  end

  def internal_developer_commands_paths
    find_commands HOMEBREW_DEV_CMD_PATH
  end

  def official_external_commands_paths
    %w[bundle services].map do |cmd|
      tap = Tap.fetch("Homebrew/#{cmd}")
      tap.install unless tap.installed?
      external_ruby_v2_cmd_path(cmd)
    end
  end

  def internal_commands
    find_internal_commands(HOMEBREW_CMD_PATH).map(&:to_s)
  end

  def internal_developer_commands
    find_internal_commands(HOMEBREW_DEV_CMD_PATH).map(&:to_s)
  end

  def internal_commands_aliases
    HOMEBREW_INTERNAL_COMMAND_ALIASES.keys
  end

  def find_internal_commands(path)
    find_commands(path).map(&:basename)
                       .map(&method(:basename_without_extension))
  end

  def external_commands
    Tap.cmd_directories.flat_map do |path|
      find_commands(path).select(&:executable?)
                         .map(&method(:basename_without_extension))
                         .map { |p| p.to_s.sub(/^brew(cask)?-/, '\1 ').strip }
    end.map(&:to_s)
       .sort
  end

  def basename_without_extension(path)
    path.basename(path.extname)
  end

  def find_commands(path)
    Pathname.glob("#{path}/*")
            .select(&:file?)
            .sort
  end
end
