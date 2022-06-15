# typed: false
# frozen_string_literal: true

require "language/python"

# A formula's caveats.
#
# @api private
class Caveats
  extend Forwardable

  attr_reader :f

  def initialize(f)
    @f = f
  end

  def caveats
    caveats = []
    begin
      build = f.build
      f.build = Tab.for_formula(f)
      s = f.caveats.to_s
      caveats << "#{s.chomp}\n" unless s.empty?
    ensure
      f.build = build
    end
    caveats << keg_only_text

    valid_shells = [:bash, :zsh, :fish].freeze
    current_shell = Utils::Shell.preferred || Utils::Shell.parent
    shells = if current_shell.present? &&
                (shell_sym = current_shell.to_sym) &&
                valid_shells.include?(shell_sym)
      [shell_sym]
    else
      valid_shells
    end
    shells.each do |shell|
      caveats << function_completion_caveats(shell)
    end

    caveats << service_caveats
    caveats << elisp_caveats
    caveats.compact.join("\n")
  end

  delegate [:empty?, :to_s] => :caveats

  def keg_only_text(skip_reason: false)
    return unless f.keg_only?

    s = if skip_reason
      ""
    else
      <<~EOS
        #{f.name} is keg-only, which means it was not symlinked into #{HOMEBREW_PREFIX},
        because #{f.keg_only_reason.to_s.chomp}.
      EOS
    end.dup

    if f.bin.directory? || f.sbin.directory?
      s << <<~EOS

        If you need to have #{f.name} first in your PATH, run:
      EOS
      s << "  #{Utils::Shell.prepend_path_in_profile(f.opt_bin.to_s)}\n" if f.bin.directory?
      s << "  #{Utils::Shell.prepend_path_in_profile(f.opt_sbin.to_s)}\n" if f.sbin.directory?
    end

    if f.lib.directory? || f.include.directory?
      s << <<~EOS

        For compilers to find #{f.name} you may need to set:
      EOS

      s << "  #{Utils::Shell.export_value("LDFLAGS", "-L#{f.opt_lib}")}\n" if f.lib.directory?

      s << "  #{Utils::Shell.export_value("CPPFLAGS", "-I#{f.opt_include}")}\n" if f.include.directory?

      if which("pkg-config", ORIGINAL_PATHS) &&
         ((f.lib/"pkgconfig").directory? || (f.share/"pkgconfig").directory?)
        s << <<~EOS

          For pkg-config to find #{f.name} you may need to set:
        EOS

        if (f.lib/"pkgconfig").directory?
          s << "  #{Utils::Shell.export_value("PKG_CONFIG_PATH", "#{f.opt_lib}/pkgconfig")}\n"
        end

        if (f.share/"pkgconfig").directory?
          s << "  #{Utils::Shell.export_value("PKG_CONFIG_PATH", "#{f.opt_share}/pkgconfig")}\n"
        end
      end
    end
    s << "\n"
  end

  private

  def keg
    @keg ||= [f.prefix, f.opt_prefix, f.linked_keg].map do |d|
      Keg.new(d.resolved_path)
    rescue
      nil
    end.compact.first
  end

  def function_completion_caveats(shell)
    return unless keg
    return unless which(shell.to_s, ORIGINAL_PATHS)

    completion_installed = keg.completion_installed?(shell)
    functions_installed = keg.functions_installed?(shell)
    return if !completion_installed && !functions_installed

    installed = []
    installed << "completions" if completion_installed
    installed << "functions" if functions_installed

    root_dir = f.keg_only? ? f.opt_prefix : HOMEBREW_PREFIX

    case shell
    when :bash
      <<~EOS
        Bash completion has been installed to:
          #{root_dir}/etc/bash_completion.d
      EOS
    when :zsh
      <<~EOS
        zsh #{installed.join(" and ")} have been installed to:
          #{root_dir}/share/zsh/site-functions
      EOS
    when :fish
      fish_caveats = +"fish #{installed.join(" and ")} have been installed to:"
      fish_caveats << "\n  #{root_dir}/share/fish/vendor_completions.d" if completion_installed
      fish_caveats << "\n  #{root_dir}/share/fish/vendor_functions.d" if functions_installed
      fish_caveats.freeze
    end
  end

  def elisp_caveats
    return if f.keg_only?
    return unless keg
    return unless keg.elisp_installed?

    <<~EOS
      Emacs Lisp files have been installed to:
        #{HOMEBREW_PREFIX}/share/emacs/site-lisp/#{f.name}
    EOS
  end

  def service_caveats
    return if !f.plist && !f.service? && !keg&.plist_installed?

    s = []

    command = if f.service?
      f.service.manual_command
    else
      f.plist_manual
    end

    return <<~EOS if !which("launchctl") && f.plist
      #{Formatter.warning("Warning:")} #{f.name} provides a launchd plist which can only be used on macOS!
      You can manually execute the service instead with:
        #{command}
    EOS

    # Brew services only works with these two tools
    return <<~EOS if !which("systemctl") && !which("launchctl") && f.service?
      #{Formatter.warning("Warning:")} #{f.name} provides a service which can only be used on macOS or systemd!
      You can manually execute the service instead with:
        #{command}
    EOS

    is_running_service = f.service? && quiet_system("ps aux | grep #{f.service.command&.first}")
    if is_running_service || (f.plist && quiet_system("/bin/launchctl list #{f.plist_name} &>/dev/null"))
      s << "To restart #{f.full_name} after an upgrade:"
      s << "  #{f.plist_startup ? "sudo " : ""}brew services restart #{f.full_name}"
    elsif f.plist_startup
      s << "To start #{f.full_name} now and restart at startup:"
      s << "  sudo brew services start #{f.full_name}"
    else
      s << "To start #{f.full_name} now and restart at login:"
      s << "  brew services start #{f.full_name}"
    end

    if f.plist_manual || f.service?
      s << "Or, if you don't want/need a background service you can just run:"
      s << "  #{command}"
    end

    # pbpaste is the system clipboard tool on macOS and fails with `tmux` by default
    # check if this is being run under `tmux` to avoid failing
    if ENV["HOMEBREW_TMUX"] && !quiet_system("/usr/bin/pbpaste")
      s << "" << "WARNING: brew services will fail when run under tmux."
    end

    "#{s.join("\n")}\n" unless s.empty?
  end
end
