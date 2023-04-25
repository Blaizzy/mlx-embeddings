# typed: true
# frozen_string_literal: true

class Module
  include T::Sig

  def attr_rw(*attrs)
    attrs.each do |attr|
      module_eval <<-EOS, __FILE__, __LINE__+1
        def #{attr}(val=nil)                         # def prefix(val=nil)
          if val.nil?                                #   if val.nil?
            if instance_variable_defined?(:@#{attr}) #      if instance_variable_defined?(:@prefix)
              return @#{attr}                        #        return @prefix
            else                                     #      else
              return nil                             #        return nil
            end                                      #      end
          end                                        #    end
                                                     #
          @#{attr} = val                             #   @prefix = val
        end                                          # end
      EOS
    end
  end
end
